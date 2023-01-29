use hex;
use std::fs::File;
use std::io;
use std::io::Read;

use crate::object::blob;

pub fn hash_object(path: String) -> String {
    let file = File::open(path);
    let mut buf = Vec::new();
    let _ = file.unwrap().read_to_end(&mut buf);

    let blob = blob::Blob::from(&buf).ok_or(io::Error::from(io::ErrorKind::InvalidInput));
    match blob {
        Ok(blob) => {
            hex::encode(blob.calc_hash())
        }
        Err(blob) => {
            blob.to_string()
        }
    }
}
