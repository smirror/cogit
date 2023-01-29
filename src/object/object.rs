use sha1::{Sha1, Digest};
use crate::object::object_type;

#[derive(Debug, Clone)]
pub struct Object {
    Hash: Sha1,
    Type: Type,
    Size: i8,
    Data: [byte],
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Hash:{}, Type:{}, Size:{}, Data", self.Hash, self.Type, self.Size, self.Data)
    }
}

impl Object {
    fn Header(&self) -> &[byte] {
        return [byte] = format!("{} {}\x00", self.Type, self.Data).into_bytes();
    }
}
