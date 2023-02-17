use std::{env, fs};
use std::fs::File;
use std::io::Write;
use std::path::Path;


pub fn init() -> String {
    let check_dir = Path::new("./.cogit").is_dir();

    // mkdir .cogit
    match fs::create_dir("./.cogit") {
        Ok(_) => {}
        Err(_) => {
    };

    // mkdir .cogit/refs
    fs::create_dir("./.cogit/refs").ok();
    fs::create_dir("./.cogit/refs/heads").ok();
    fs::create_dir("./.cogit/refs/tags").ok();

    // mkdir .cogit/objects
    fs::create_dir("./.cogit/objects").ok();
    fs::create_dir("./.cogit/objects/info").ok();
    fs::create_dir("./.cogit/objects/pack").ok();

    // mkdir .cogit/info
    fs::create_dir("./.cogit/info").ok();
    if !Path::new("./.cogit/info/exclude").is_file() {
        let file = File::open("./.cogit/info/exclude");
        match file {
            Ok(mut file) => {
                file.write_all(String::from("[core]\n\trepositoryformatversion = 0\n\tfilemode = true\n\tbare = false\n\tlogallrefupdates = true").as_bytes()).unwrap();
            }
            Err(_) => {}
        }
    }

    let path = env::current_dir();
    if check_dir {
        format!("Reinitialized existing Git repository in {}/.cogit!!", path.unwrap().to_string_lossy())
    } else {
        format!("Initialized existing Git repository in {}/.cogit!!!", path.unwrap().to_string_lossy())
    }
}
