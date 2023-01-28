enum Type {
    UndefinedObject,
    CommitObject,
    TreeObject,
    BlobObject,
    TagObject,
}

impl Type {
    fn string(&self) -> String {
        let objectTypeString: [&str; 5] = ["undefined", "commit", "tree", "blob", "tag"];
        return objectTypeString.index(self.type_id()).to_string();
    }
}


fn NewType(typeString: &str) -> Type {
    return match typeString {
        "commit" => { Type::CommitObject }
        "tree" => { Type::TreeObject }
        "blob" => { Type::BlobObject }
        "tag" => { Type::TagObject }
        _ => { Type::UndefinedObject }
    };
}
