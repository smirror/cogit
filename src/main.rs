pub mod cmd;
pub mod object;

extern crate exitcode;

use std::env;
use std::ops::Index;
use std::process::exit;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() == 1 {
        // TODO: add help messages
        println!("plz subcommand.");
        exit(exitcode::OK)
    }

    match args.index(1).as_str() {
        "add" => {
            println!("add")
        }
        "cat-file" => {
            println!("cat-file")
        }
        "commit" => {
            println!("init")
        }
        "diff" => {
            println!("diff")
        }
        "init" => {
            println!("{}", cmd::init::init())
        }
        "status" => {
            println!("status")
        }
        "log" => {
            println!("log")
        }
        "hash-object" => {
            println!(
                "{}",
                cmd::hash_object::hash_object(args.get(2).unwrap().clone())
            )
        }
        _ => {
            println!("sorry. not {} subcommand.", args.index(1));
        }
    }
}
