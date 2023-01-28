use sha1::{Sha1, Digest};

struct Object {
    Hash: Sha1,
    Type: Type,
    Size: i8,
    Data: [byte],
}

impl Object {
    fn Header(&self) -> &[byte] {
        return [byte] = format!("{} {}\x00", self.Type, self.Data).into_bytes();
    }
}
