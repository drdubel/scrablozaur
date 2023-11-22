use pyo3::prelude::*;
use serde;
use std::collections::HashMap;
use std::fmt::{Display, Formatter, Result};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader};

static mut NEXT_ID: i32 = 0;

#[derive(Debug, Clone, Eq)]
#[pyclass(get_all, set_all, subclass)]
struct Node {
    children: HashMap<char, Node>,
    is_terminal: bool,
    id: i32,
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let mut out: Vec<String> = Vec::new();
        if self.is_terminal {
            out.push("1".to_string());
        } else {
            out.push("0".to_string());
        }
        for (key, val) in &self.children {
            out.push(key.to_string());
            out.push(val.id.to_string());
        }

        write!(f, "{}", out.join("_"))
    }
}

#[pymethods]
impl Node {
    #[new]
    fn new() -> Self {
        unsafe { NEXT_ID += 1 };
        Node {
            children: HashMap::new(),
            is_terminal: false,
            id: unsafe { NEXT_ID - 1 },
        }
    }
    fn __repr__(&self) -> String {
        format!("{}", self)
    }
}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.__repr__().hash(state);
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool {
        self.__repr__() == other.__repr__()
    }
}

#[pyfunction]
fn length_common_prefix(prev_word: &str, word: &str) -> i32 {
    let mut pref_len = 0;
    for (char1, char2) in prev_word.chars().zip(word.chars()) {
        if char1 != char2 {
            return pref_len;
        }
        pref_len += 1;
    }
    pref_len
}

#[pyfunction]
fn read_dawg_from_file(path: &str) -> Node {
    let file = File::open(path).expect("Failed to open file");
    let reader = BufReader::new(file);
    let data: Pickle = Pickle::new(reader).expect("Failed to parse pickle");
    let my_value = data.get::<Node>().expect("Failed to get data");
    dawg
}

#[pymodule]
fn dawgpyrust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Node>()?;
    m.add_function(wrap_pyfunction!(length_common_prefix, m)?)?;
    m.add_function(wrap_pyfunction!(read_dawg_from_file, m)?)?;
    Ok(())
}
