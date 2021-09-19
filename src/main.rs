//! Design specification for our yet-to-be-named SPSC disk buffer implementation:
//!
//! We provide a single writer/single reader interface to an underlying set of files that
//! conceptually represent a ring buffer.  Unlike a typical ring buffer, we block writes when the
//! total size of all unread records reaches the configured limit.  It may be possible to alter the
//! design in the future such that we can provide a "drop oldest" operation mode, but that is
//! out-of-scope for version 1 of this design.
//!
//! Design constraints / invariants:
//! - buffer can be a maximum of 2TB in total size
//! - data files do not exceed 128MB
//! - files are allocated upfront (each file's full 128MB is allocated when created)
//! - all headers (ledger, data file) are written in network byte order (big endian) when integers
//!   are involved
//!
//! At a high-level, records that are written end up in one of many underlying data files, while the
//! ledgber file -- number of records, writer and reader positions, etc -- is stored in a separate
//! file.  Data files function primarily with a "last process who touched it" ownership model: the
//! writer always creates new files, and the reader deletes files when they have been fully read.
//!
//! Internally, data files consist of a simplified structure that is optimized for the ring buffer
//! use case.  Records are packed together with a minimalistic layout:
//!
//!   buffer-data-001.dat:
//!     [data file record count - unsigned 64-bit integer]
//!     [data file size - unsigned 64-bit integer]
//!     [record ID - unsigned 64-bit integer]
//!     [record length - unsigned 64-bit integer]
//!     [record data - N bytes]
//!
//! The record ID/length/data superblocks repeat infinitely until adding another would exceed the
//! configured data file size limit, in which case a new data file is started. A record cannot
//! exceed the maximum size of a data file.  Attempting to buffer such a record will result in an error.
//!
//! Likewise, the bookkeeping file consists of a simplified structure that is optimized for being
//! shared via a memory-mapped file interface between the writer and reader:
//!
//!   buffer.db:
//!     [total record count - unsigned 64-bit integer]
//!     [total buffer size - unsigned 64-bit integer]
//!     [next record ID - unsigned 64-bit integer]
//!     [writer current data file ID - unsigned 16-bit integer]
//!     [reader current data file ID - unsigned 16-bit integer]
//!     [reader last record ID - unsigned 64-bit integer]
//!
//! As this buffer is meant to emulate a ring buffer, most of the bookkeeping resolves around the
//! writer and reader being able to quickly figure out where they left off.  Record and data file
//! IDs are simply rolled over when they reach the maximum of their data type, and are incremented
//! indiscriminately rather than reused if one is retired within the 0 - N range.
//!
//! TODO: think through whether or not we can use total file size to ensure that we never try to
//! open more than 4096 files (2TB max buffer size / 256MB max data file size) total, so that we can
//! avoid needing an array/bitmap/etc tracking which files are in use.

use std::{
    convert::TryInto,
    default,
    fmt::Write,
    fs::OpenOptions,
    io, mem,
    ops::Deref,
    path::{Path, PathBuf},
    ptr, slice,
    sync::{
        atomic::{AtomicU16, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use memmap2::{MmapMut, MmapOptions};
use parking_lot::{Mutex, MutexGuard};

const LEDGER_FILE_SIZE: usize = 36;
const DATA_FILE_SIZE: usize = 64 * 1024 * 1024;
const DATA_FILE_HEADER_SIZE: usize = 16;
const DATA_FILE_RECORD_DATA_MAX_USABLE: usize = DATA_FILE_SIZE - DATA_FILE_HEADER_SIZE;
const DATA_FILE_RECORD_HEADER_SIZE: usize = 16;
const DATA_FILE_MAX_RECORD_SIZE: usize =
    DATA_FILE_SIZE - DATA_FILE_HEADER_SIZE - DATA_FILE_RECORD_HEADER_SIZE;

#[derive(Debug, Default)]
struct LedgerState {
    // Total number of records persisted in this buffer.
    total_records: AtomicU64,
    // Total size of all data files used by this buffer.
    total_buffer_size: AtomicU64,
    // Next record ID to use when writing a record.
    next_record_id: AtomicU64,
    // The current data file ID being written to.
    writer_current_data_file_id: AtomicU16,
    // The current data file ID being read from.
    reader_current_data_file_id: AtomicU16,
    // The last record ID read by the reader.
    reader_last_record_id: AtomicU64,
}

impl LedgerState {
    pub fn allocate_record_id(&self) -> u64 {
        self.next_record_id.fetch_add(1, Ordering::Acquire)
    }

    /// Gets the current write file ID.
    pub fn current_writer_file_id(&self) -> u16 {
        self.writer_current_data_file_id.load(Ordering::Acquire)
    }

    /// Increments the current writer file ID.
    pub fn increment_writer_file_id(&self) {
        self.writer_current_data_file_id
            .fetch_add(1, Ordering::AcqRel);
    }

    pub fn serialize_to(&self, dst: &mut [u8]) {
        // CLARITY TODO: This is very ugly, and fragile due to field offsets.  It'd be nice if we
        // had a macro or something to make this a little more programmatic/repeatable/machine
        // checkable.  Given that we only have three structs which we serialize in this fashion,
        // though, that could be overkill.
        //
        // PERFORMANCE TODO: This is a nice, safe variant of pushing the state into the data file,
        // but I'm not sure if doing a pointer-level `memcpy` action would be meaningfully faster.
        let total_records = self.total_records.load(Ordering::SeqCst).to_be_bytes();
        let total_buffer_size = self.total_buffer_size.load(Ordering::SeqCst).to_be_bytes();
        let next_record_id = self.next_record_id.load(Ordering::SeqCst).to_be_bytes();
        let writer_current_data_file_id = self
            .writer_current_data_file_id
            .load(Ordering::SeqCst)
            .to_be_bytes();
        let reader_current_data_file_id = self
            .reader_current_data_file_id
            .load(Ordering::SeqCst)
            .to_be_bytes();
        let reader_last_record_id = self
            .reader_last_record_id
            .load(Ordering::SeqCst)
            .to_be_bytes();

        let mut src = Vec::new();
        src.extend_from_slice(&total_records[..]);
        src.extend_from_slice(&total_buffer_size[..]);
        src.extend_from_slice(&next_record_id[..]);
        src.extend_from_slice(&writer_current_data_file_id[..]);
        src.extend_from_slice(&reader_current_data_file_id[..]);
        src.extend_from_slice(&reader_last_record_id[..]);

        debug_assert!(dst.len() == LEDGER_FILE_SIZE);
        debug_assert!(src.len() == LEDGER_FILE_SIZE);
        dst.copy_from_slice(&src[..]);
    }

    pub fn deserialize_from(&mut self, src: &[u8]) {
        // CLARITY TODO: This is very ugly, and fragile due to field offsets.  It'd be nice if we
        // had a macro or something to make this a little more programmatic/repeatable/machine
        // checkable.  Given that we only have three structs which we serialize in this fashion,
        // though, that could be overkill.
        debug_assert!(src.len() == LEDGER_FILE_SIZE);

        self.total_records = src[..8]
            .try_into()
            .map(u64::from_be_bytes)
            .map(AtomicU64::new)
            .expect("should have had 8 bytes");
        self.total_buffer_size = src[8..16]
            .try_into()
            .map(u64::from_be_bytes)
            .map(AtomicU64::new)
            .expect("should have had 8 bytes");
        self.next_record_id = src[16..24]
            .try_into()
            .map(u64::from_be_bytes)
            .map(AtomicU64::new)
            .expect("should have had 8 bytes");
        self.writer_current_data_file_id = src[24..26]
            .try_into()
            .map(u16::from_be_bytes)
            .map(AtomicU16::new)
            .expect("should have had 2 bytes");
        self.reader_current_data_file_id = src[26..28]
            .try_into()
            .map(u16::from_be_bytes)
            .map(AtomicU16::new)
            .expect("should have had 2 bytes");
        self.reader_last_record_id = src[28..36]
            .try_into()
            .map(u64::from_be_bytes)
            .map(AtomicU64::new)
            .expect("should have had 8 bytes");
    }
}

#[derive(Debug)]
struct Ledger {
    // Path to the data directory.
    data_dir: PathBuf,
    // Handle to the memory-mapped ledger file.
    ledger_mmap: Mutex<MmapMut>,
    // Ledger state.
    state: LedgerState,
}

impl Ledger {
    pub fn allocate_record_id(&self) -> u64 {
        self.state.next_record_id.fetch_add(1, Ordering::Acquire)
    }

    pub fn get_writer_current_data_file_path(&self) -> PathBuf {
        self.get_data_file_path(self.state.current_writer_file_id())
    }

    pub fn increment_writer_file_id(&self) {
        self.state.increment_writer_file_id();
    }

    pub fn get_reader_current_data_file_path(&self) -> PathBuf {
        self.get_data_file_path(
            self.state
                .reader_current_data_file_id
                .load(Ordering::Acquire),
        )
    }

    pub fn get_data_file_path(&self, file_id: u16) -> PathBuf {
        self.data_dir.join(format!("buffer-data-{}.dat", file_id))
    }

    pub fn state(&self) -> &LedgerState {
        &self.state
    }

    pub fn data_dir(&self) -> &PathBuf {
        &self.data_dir
    }

    fn load_inner(&mut self) -> io::Result<()> {
        // TODO: this theoretically doesn't need to return a Result right now, let alone an
        // io::Result, but at some point we should likely be adding checksums and doing other
        // checks, so loading our state from disk would become a fallible operation

        // INVARIANT: We always create the ledger file with a size of LEDGER_FILE_SIZE, which never
        // changes over time.  We can be sure that the slice we take to read the ledger state will
        // be exactly LEDGER_FILE_SIZE bytes.
        let ledger_mmap = self.ledger_mmap.lock();
        let ledger_region = &ledger_mmap[..];
        debug_assert_eq!(ledger_region.len(), LEDGER_FILE_SIZE);
        self.state.deserialize_from(ledger_region);

        Ok(())
    }

    pub fn track_write(&self, bytes_written: u64) {
        self.state.total_records.fetch_add(1, Ordering::Release);
        self.state
            .total_buffer_size
            .fetch_add(bytes_written, Ordering::Release);
    }

    pub fn flush(&self) -> io::Result<()> {
        // INVARIANT: We always create the ledger file with a size of LEDGER_FILE_SIZE, which never
        // changes over time.  We can be sure that the slice we take to write the ledger state will
        // be exactly LEDGER_FILE_SIZE bytes.
        let mut ledger_mmap = self.ledger_mmap.lock();
        let ledger_region = &mut ledger_mmap[..];
        debug_assert_eq!(ledger_region.len(), LEDGER_FILE_SIZE);
        self.state.serialize_to(ledger_region);

        ledger_mmap.flush()
    }

    pub fn load_or_create<P>(data_dir: P) -> io::Result<Ledger>
    where
        P: AsRef<Path>,
    {
        let ledger_path = data_dir.as_ref().join("buffer.db");
        let ledger_handle = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&ledger_path)?;
        let _ = ledger_handle.set_len(LEDGER_FILE_SIZE as u64)?;

        let ledger_mmap = unsafe { MmapOptions::new().map_mut(&ledger_handle)? };
        let mut ledger = Ledger {
            data_dir: data_dir.as_ref().to_owned(),
            ledger_mmap: Mutex::new(ledger_mmap),
            state: LedgerState::default(),
        };

        let _ = ledger.load_inner()?;
        Ok(ledger)
    }
}

#[derive(Clone, Default)]
struct DataFileState {
    data_file_records: u64,
    data_file_size: u64,
}

impl DataFileState {
    pub fn track_write(&mut self, bytes_written: u64) {
        self.data_file_records += 1;
        self.data_file_size += bytes_written;
    }

    pub fn can_write(&self, record_len: u64) -> bool {
        // INVARIANT: The data file size can never be greater than DATA_FILE_RECORD_DATA_MAX_USABLE,
        // and so we know that subtracting `current_size` from DATA_FILE_RECORD_DATA_MAX_USABLE will
        // never underflow.
        let record_data_max_usable = DATA_FILE_RECORD_DATA_MAX_USABLE as u64;
        let current_size = self.data_file_size;
        assert!(record_data_max_usable >= current_size);

        // INVARIANT: The writer doesn't allow records that are bigger the maximum data file size,
        // less the data file header size and record header size.  These limits are far, far below
        // the maximum size of a u64, so `bytes_to_write` + DATA_FILE_RECORD_HEADER_SIZE should
        // never overflow a u64.
        let free_space = record_data_max_usable - current_size;
        free_space >= record_len + DATA_FILE_RECORD_HEADER_SIZE as u64
    }

    pub fn reset(&mut self) {
        self.data_file_records = 0;
        self.data_file_size = 0;
    }

    pub fn serialize_to(&self, dst: &mut [u8]) {
        // CLARITY TODO: This is very ugly, and fragile due to field offsets.  It'd be nice if we
        // had a macro or something to make this a little more programmatic/repeatable/machine
        // checkable.  Given that we only have three structs which we serialize in this fashion,
        // though, that could be overkill.
        //
        // PERFORMANCE TODO: This is a nice, safe variant of pushing the state into the data file,
        // but I'm not sure if doing a pointer-level `memcpy` action would be meaningfully faster.
        let data_file_records = self.data_file_records.to_be_bytes();
        let data_file_size = self.data_file_size.to_be_bytes();

        let mut src = Vec::new();
        src.extend_from_slice(&data_file_records[..]);
        src.extend_from_slice(&data_file_size[..]);

        debug_assert_eq!(dst.len(), DATA_FILE_HEADER_SIZE);
        debug_assert_eq!(src.len(), DATA_FILE_HEADER_SIZE);
        dst.copy_from_slice(&src[..]);
    }

    pub fn deserialize_from(&mut self, src: &[u8]) {
        // CLARITY TODO: This is very ugly, and fragile due to field offsets.  It'd be nice if we
        // had a macro or something to make this a little more programmatic/repeatable/machine
        // checkable.  Given that we only have three structs which we serialize in this fashion,
        // though, that could be overkill.
        debug_assert_eq!(src.len(), DATA_FILE_HEADER_SIZE);

        self.data_file_records = src[..8]
            .try_into()
            .map(u64::from_be_bytes)
            .expect("should have had 8 bytes");
        self.data_file_size = src[8..16]
            .try_into()
            .map(u64::from_be_bytes)
            .expect("should have had 8 bytes");
    }
}

#[repr(C)]
struct RecordHeader {
    // ID of this record.
    record_id: u64,
    // Length of this record in bytes.
    record_length: u64,
}

impl RecordHeader {
    pub fn serialize_to(&self, dst: &mut [u8]) {
        // CLARITY TODO: This is very ugly, and fragile due to field offsets.  It'd be nice if we
        // had a macro or something to make this a little more programmatic/repeatable/machine
        // checkable.  Given that we only have three structs which we serialize in this fashion,
        // though, that could be overkill.
        //
        // PERFORMANCE TODO: This is a nice, safe variant of pushing the header into the data file,
        // but I'm not sure if doing a pointer-level `memcpy` action would be meaningfully faster.
        let record_id = self.record_id.to_be_bytes();
        let record_length = self.record_length.to_be_bytes();

        let mut src = Vec::new();
        src.extend_from_slice(&record_id[..]);
        src.extend_from_slice(&record_length[..]);

        debug_assert_eq!(dst.len(), DATA_FILE_RECORD_HEADER_SIZE);
        debug_assert_eq!(src.len(), DATA_FILE_RECORD_HEADER_SIZE);
        dst.copy_from_slice(&src[..]);
    }

    pub fn deserialize_from(&mut self, src: &[u8]) {
        // CLARITY TODO: This is very ugly, and fragile due to field offsets.  It'd be nice if we
        // had a macro or something to make this a little more programmatic/repeatable/machine
        // checkable.  Given that we only have three structs which we serialize in this fashion,
        // though, that could be overkill.
        debug_assert_eq!(src.len(), DATA_FILE_RECORD_HEADER_SIZE);

        self.record_id = src[..8]
            .try_into()
            .map(u64::from_be_bytes)
            .expect("should have had 8 bytes");
        self.record_length = src[8..16]
            .try_into()
            .map(u64::from_be_bytes)
            .expect("should have had 8 bytes");
    }
}

struct WriteState {
    ledger: Arc<Ledger>,
    data_file: Option<MmapMut>,
    data_file_state: DataFileState,
}

impl From<Arc<Ledger>> for WriteState {
    fn from(ledger: Arc<Ledger>) -> Self {
        WriteState {
            ledger,
            data_file: None,
            data_file_state: DataFileState::default(),
        }
    }
}

struct WriteGuard<'a> {
    inner: MutexGuard<'a, WriteState>,
}

impl<'a> WriteGuard<'a> {
    pub fn write<R>(&mut self, record: R) -> io::Result<()>
    where
        R: AsRef<[u8]>,
    {
        let buf = record.as_ref();
        let buf_len = buf.len();

        // Check that the record isn't bigger than the maximum record size.  We'll do a more
        // specific check related to whatever current data file we have open is, and what it has for
        // remaining size, to ensure that we open a new data file if need be.
        if buf_len > DATA_FILE_MAX_RECORD_SIZE {
            return Err(io::Error::new(io::ErrorKind::Other, "record too large"));
        }

        // Ensure we have enough free space in the current file to write the record, or close the
        // current file and open a new one.
        //
        // TODO: Add support for splitting a record across multiple files, since otherwise we'll
        // end up losing space at the end when we're simply a few bytes short of being able to fit.
        //
        // We can technically already determine, based on the record length, whether or not the
        // current data file could possibly contain the record data itself, although the hard part
        // to figure out is if we rewrite the same record header in the next file, and whether or
        // not we count a record that spans multiple data files as a record in both, in terms of
        // `total_records`.
        //
        // We could likely do something like... if record length is N, and N + current data file
        // offset is > max file size, then we know we're spilling over to another file, and we
        // write a normal record block to the next data file, with the same record ID, but subtract
        // what we've already written from record length for that 2nd, or 3rd, or nth record block.
        // This way we know we're still reading the same record, and the record length always tells
        // us how much is remaining, so we can keep carrying over and figure out if we need to read
        // from _another_ data file.
        //
        // Where this potentially gets _very_ tricky is that if we already had the maximum number of
        // files minus one, let's say, so we do our first partial write into the current (last
        // possible, mind you) data file, and then need to spill over to another one... we'd
        // technically be blocked waiting for the reader to read that very first data file and
        // remove it, so we were able to recreate it.
        //
        // Technically, we _could_ do something like figuring out how many possible data files we
        // could instantaneously allocate based on the reader's progress, and use that to figure out
        // the maximum record we could possibly write at that moment, and block if we didn't have
        // the capacity to do so.  The math itself is simple because it's just a little extra
        // arithmetic to consider the remaining data files that we can write, but it is extra logic
        // that might become a little too thorny.
        //
        // The other option is that we simply keep things as-is, and don't allow writing a record
        // bigger than a single data file and that we don't bother trying to spill over to a new
        // data file, simply preferring to open a new data file.  This isn't the most efficient use
        // of space but it _is_ simpler.
        //
        // ASYNC TODO: This would be an asynchronous wait point to return from if we didn't have
        // enough capacity for this write. Specifically, we'd be waiting here because the next file we
        // want to write to already exists and is pending being read by the reader.
        let _ = self.ensure_capacity(buf_len)?;

        let record_id = self.inner.ledger.allocate_record_id();

        // INVARIANT: The data file size is limited to a maximum of DATA_FILE_SIZE, which is far
        // below even 2^32, so casting `data_file_size` to to u64 is safe and will not truncate.
        // Additionally, adding DATA_FILE_HEADER_SIZE to it will not overflow a usize even on 32-bit
        // platforms.
        let write_offset =
            self.inner.data_file_state.data_file_size as usize + DATA_FILE_HEADER_SIZE;
        let header_start = write_offset;
        let header_end = header_start + DATA_FILE_RECORD_HEADER_SIZE;
        let data_start = header_end;
        let data_end = data_start + buf_len;

        // Write our record header and data.
        let data_file_mmap = self
            .inner
            .data_file
            .as_mut()
            .expect("mmaped dat file should exist");
        let record_header = RecordHeader {
            record_id,
            record_length: buf.len() as u64,
        };
        let record_region = &mut data_file_mmap[header_start..header_end];
        record_header.serialize_to(record_region);
        let data_region = &mut data_file_mmap[data_start..data_end];
        data_region.copy_from_slice(buf);

        // Update the metadata now that we've written the record.
        let bytes_written = (buf_len + DATA_FILE_RECORD_HEADER_SIZE) as u64;
        self.inner.ledger.track_write(bytes_written);
        self.inner.data_file_state.track_write(bytes_written);

        Ok(())
    }

    fn ensure_capacity(&mut self, bytes_to_write: usize) -> io::Result<()> {
        // If we have an open data file, check to see if it has enough room for us to write into.
        // If not, we make sure it's flushed to disk, bump our writer file ID, and let the "no open
        // file" logic below take over.
        if self.inner.data_file.is_some() {
            if self.inner.data_file_state.can_write(bytes_to_write as u64) {
                return Ok(());
            } else {
                // Not enough space to write the record, so we need to close out the current
                // data file and open a new one.  Flush all of our current metadata and the
                // data file itself before opening a new one.
                let _ = self.flush()?;
                self.inner.data_file = None;

                self.inner.ledger.increment_writer_file_id();
            }
        }

        // We currently have no open data file, or we didn't have enough space in the
        // previously-opened one, either way, create a new one, and set it up.
        //
        // TODO: How do we detect (correctly) that, when the reader says it's reading the same file
        // ID as we're about to create here, that we're actually just picking up where we left off
        // with a not-yet-full data file _or_ that we've looped around with file IDs and are
        // actually trying to write to a file that is full and not yet read yet?
        //
        // I suspect what it will come down to is reading the data file state initially and seeing
        // if this file is brand new (0 records, 0 size) or not.
        //
        // ASYNC TODO: This would be an asynchronous wait point to return from if we didn't have
        // enough capacity for this write. Specifically, we'd be waiting here because the file we
        // want to write to already exists and is pending being read by the reader.
        let data_file_path = self.inner.ledger.get_writer_current_data_file_path();
        let data_file_handle = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&data_file_path)?;
        let _ = data_file_handle.set_len(DATA_FILE_SIZE as u64)?;

        let data_file_mmap = unsafe { MmapOptions::new().map_mut(&data_file_handle)? };
        self.inner.data_file = Some(data_file_mmap);
        self.inner.data_file_state.reset();

        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        // INVARIANT: We always created data files with a size of DATA_FILE_SIZE, which we assert to
        // always be larger than DATA_FILE_HEADER_SIZE, so our `state_region` slice will always be
        // valid and of DATA_FILE_HEADER_SIZE bytes long.
        let state = self.inner.data_file_state.clone();
        if let Some(data_file_mmap) = self.inner.data_file.as_mut() {
            let state_region = &mut data_file_mmap[..DATA_FILE_HEADER_SIZE];
            state.serialize_to(state_region);
            let _ = data_file_mmap.flush()?;
        }

        self.inner.ledger.flush()
    }

    pub fn commit(mut self) -> io::Result<()> {
        self.flush()
    }
}

struct Writer {
    ledger: Arc<Ledger>,
    state: Mutex<WriteState>,
}

impl Writer {
    pub fn new(ledger: Arc<Ledger>) -> Self {
        let state = WriteState::from(Arc::clone(&ledger));
        Writer {
            ledger,
            state: Mutex::new(state),
        }
    }

    pub fn transaction(&self) -> WriteGuard<'_> {
        WriteGuard {
            inner: self.state.lock(),
        }
    }
}

struct Reader {
    ledger: Arc<Ledger>,
    data_file: Option<MmapMut>,
}

struct Buffer;

impl Buffer {
    pub fn from_path<P>(data_dir: P) -> io::Result<(Writer, Reader)>
    where
        P: AsRef<Path>,
    {
        // Establishing invariants about the size of various structures within the ledger and data
        // files, to make life easier when implementing and ensuring operations are valid.
        assert!(DATA_FILE_SIZE > DATA_FILE_HEADER_SIZE);
        assert!(DATA_FILE_RECORD_DATA_MAX_USABLE < DATA_FILE_SIZE);

        let ledger = Arc::new(Ledger::load_or_create(data_dir)?);

        let writer = Writer::new(Arc::clone(&ledger));

        let reader = Reader {
            ledger,
            data_file: None,
        };

        Ok((writer, reader))
    }
}

fn main() {
    let (writer, _reader) = Buffer::from_path("/tmp/mmap-testing").expect("failed to open buffer");

    // will end up being around ~4.4kb
    let mut payload = Vec::new();
    for _ in 0..400 {
        payload.extend_from_slice(&[
            b'h', b'e', b'l', b'l', b'o', b' ', b'w', b'o', b'r', b'l', b'd',
        ]);
    }

    for i in 0..20_000 {
        let mut tx = writer.transaction();
        tx.write(&payload[..]).expect("failed to write record");
        tx.commit().expect("failed to commit transaction");

        if i % 1000 == 0 {
            println!("at txn {}", i);
        }
    }
}
