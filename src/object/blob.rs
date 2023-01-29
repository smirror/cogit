use sha1::{Digest, Sha1};
use std::fmt;

#[derive(Debug, Clone)]
pub struct Blob {
    pub size: usize,
    pub content: String,
}

impl fmt::Display for Blob {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.content)
    }
}

impl Blob {
    // 構造体を作成する関数new
    pub fn new(content: String) -> Self {
        Self {
            size: content.len(),
            content,
        }
    }

    // 変換するための関数
    pub fn from(bytes: &[u8]) -> Option<Self> {
        let content = String::from_utf8(bytes.to_vec());

        match content {
            Ok(content) => Some(Self {
                size: content.len(),
                content,
            }),
            _ => None,
        }
    }

    // 書き込むためフォーマットにするメソッド
    pub fn as_bytes(&self) -> Vec<u8> {
        let header = format!("blob {}\0", self.size);
        let store = format!("{}{}", header, self.content);

        Vec::from(store.as_bytes())
    }

    // hash値を計算するメソッド
    pub fn calc_hash(&self) -> Vec<u8> {
        Vec::from(Sha1::digest(&self.as_bytes()).as_slice())
    }
}
