use std::error;

struct ErrInvalidObject;

impl fmt::Display for ErrInvalidObject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "invalid object")
    }
}

impl error::Error for ErrInvalidObject {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

struct ErrNotCommitObject;

impl fmt::Display for ErrNotCommitObject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "not commit object")
    }
}

impl error::Error for ErrNotCommitObject {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

struct ErrInvalidCommitObject;

impl fmt::Display for ErrInvalidCommitObject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "invalid commit object")
    }
}

impl error::Error for ErrInvalidCommitObject {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}
