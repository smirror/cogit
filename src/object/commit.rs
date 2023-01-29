use chrono::{DateTime, FixedOffset, TimeZone, Utc};
#[cfg(feature = "json")]
use serde::Serialize;
use sha1::{Digest, Sha1};
use std::fmt;

#[derive(Debug, Clone)]
pub struct User {
    pub name: String,
    pub email: String,
    pub timestamp: DateTime<FixedOffset>,
}

#[derive(Debug, Clone)]
pub struct Commit {
    pub hash: sha1::SHA1,
    pub size: int,
    pub tree: sha1::SHA1,
    pub parent: Option<sha1::SHA1>,
    pub author: User,
    pub committer: User,
    pub message: String,
}

impl User {
    pub fn new(name: String, email: String, timestamp: DateTime<FixedOffset>) -> Self {
        Self {
            name,
            email,
            timestamp,
        }
    }
    pub fn from(bytes: &[u8]) -> Option<Self> {
        //bytesは"USER_NAME <EMAIL> TIME_STAMP OFFSET"のようになっている
        let name = String::from_utf8(
            bytes
                .into_iter()
                .take_while(|&&x| x != b'<')
                .map(|&x| x)
                .collect(),
        )
            .map(|x| String::from(x.trim()))
            .ok()?;

        let after_username = String::from_utf8(
            bytes
                .into_iter()
                .skip_while(|&&x| x != b'<')
                .map(|&x| x)
                .collect(),
        )
            .ok()?;

        let mut info = after_username.splitn(3, " ");

        // 1 番目の要素の < と > を削る
        let email = info
            .next()
            .map(|x| String::from(x.trim_matches(|x| x == '<' || x == '>')))?;
        // 2 番目の要素を数字に変換して TimeStamp として扱う
        let timestamp = Utc.timestamp(info.next().and_then(|x| x.parse::<i64>().ok())?, 0);

        // 3 番目の要素を数字に変換して Offset として扱う(正負で西か東かを分ける)
        let offset = info.next().and_then(|x| x.parse::<i32>().ok()).map(|x| {
            if x < 0 {
                FixedOffset::west(x / 100 * 60 * 60)
            } else {
                FixedOffset::east(x / 100 * 60 * 60)
            }
        })?;

        Some(Self::new(
            name,
            email,
            offset.from_utc_datetime(&timestamp.naive_utc()),
        ))
    }
}

impl Commit {
    pub fn new(
        hash: sha1::SHA1,
        size: int,
        tree: sha1::SHA1,
        parent: Option<sha1::SHA1>,
        author: User,
        committer: User,
        message: String,
    ) -> Self {
        Self {
            hash,
            size,
            tree,
            parent,
            author,
            committer,
            message,
        }
    }
    pub fn from(bytes: &[u8]) -> Option<Self> {
        let mut iter = bytes.split(|&c| c == b'\n').filter(|s| s != b"");

        let tree = iter
            .next()
            .map(|s| {
                s.splitn(2, |&c| c == b' ')
                    .skip(1) // 最初はtreeで決まっているので最初の要素を捨てる
                    .flatten()
                    .map(|&x| x)
                    .collect::<Vec<_>>()
            })
            .and_then(|x| String::from_utf8(x).ok())?;

        let parent = &iter
            .next()
            .map(|x| {
                x.splitn(2, |&x| x == b' ')
                    .map(Vec::from)
                    .map(|x| String::from_utf8(x).ok().unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .ok_or(Vec::new())
            .and_then(|x| match x[0].as_str() {
                "parent" => Ok(x[1].clone()), // 最初の文字列が "parent" になっていたら
                _ => Err([x[0].as_bytes(), b" ", x[1].as_bytes()].concat()), // それ以外だったら元の形式に戻してErr に包む
            });

        let author = match parent {
            Ok(_) => iter.next().map(|x| Vec::from(x)), // parent が Ok だったら iterator から取る
            Err(v) => Some(v.clone()),                  // Err だったらその値を使う
        }
            .map(|x| {
                x.splitn(2, |&x| x == b' ')
                    .skip(1)
                    .flatten()
                    .map(|&x| x)
                    .collect::<Vec<_>>()
            })
            .and_then(|x| User::from(x.as_slice()))?;

        let committer = iter
            .next()
            .map(|x| {
                x.splitn(2, |&x| x == b' ')
                    .skip(1)
                    .flatten()
                    .map(|&x| x)
                    .collect::<Vec<_>>()
            })
            .and_then(|x| User::from(x.as_slice()))?;

        let message = iter
            .next()
            .map(Vec::from)
            .and_then(|x| String::from_utf8(x).ok())?;

        Some(Self::new(
            tree,
            parent.clone().ok(),
            author,
            committer,
            message,
        ))
    }

    pub fn as_bytes(&self) -> Vec<u8> {
        let content = format!("{}", self);
        let header = format!("commit {}\0", content.len());
        let val = format!("{}{}", header, content);

        Vec::from(val.as_bytes())
    }
    pub fn calc_hash(&self) -> Vec<u8> {
        Vec::from(Sha1::digest(&self.as_bytes()).as_slice())
    }
}

impl fmt::Display for User {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} <{}> {} {:+05}",
            self.name,
            self.email,
            self.timestamp.timestamp(),
            self.timestamp.offset().local_minus_utc() / 36
        )
    }
}

impl fmt::Display for Commit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let tree = format!("tree {}", self.tree);
        let parent = self
            .parent
            .clone()
            .map(|x| format!("parent {}\n", x))
            .unwrap_or_default();
        let author = format!("author {}", self.author);
        let committer = format!("committer {}", self.committer);

        write!(
            f,
            "{}\n{}{}\n{}\n\n{}\n",
            tree, parent, author, committer, self.message
        )
    }
}
