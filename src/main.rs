extern crate exitcode;

use std::env;
use std::ops::Index;
use std::process::exit;

fn main() {
    let args:Vec<String> = env::args().collect();
    if args.len() == 1{
        // TODO: add help messages
        println!("plz subcommand.");
        exit(exitcode::OK)
    }

    match args.index(1).as_str() {
        "init" => {
            println!("init")
        }
        "add" => {
            println!("add")
        }
        "commit" => {
            println!("init")
        }
        "log" => {
            println!("log")
        }
        _ => {
            println!("sorry. not {} subcommand.", args.index(1));
        }
    }
}
