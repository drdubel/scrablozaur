use pyo3::prelude::*;

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

fn minimize<'a>(
    curr_node: Node,
    pref_len: i32,
    minimized_nodes: HashMap<&'a Node, &'a Node>,
    non_minimized_nodes: Vec<(Node, char, Node)>,
) -> (Node, HashMap<&'a Node, &'a Node>) {
    let mut new_minimized_nodes = minimized_nodes.clone();
    let mut new_curr_node = curr_node.clone();
    for i in 0..(non_minimized_nodes.len() - pref_len as usize) {
        let (parent, letter, child) =
            non_minimized_nodes[non_minimized_nodes.len() - 1 - i].clone();
        let mut new_parent = parent.clone();
        if minimized_nodes.contains_key(&child) {
            new_parent
                .children
                .insert(letter, minimized_nodes[&child].clone());
        } else {
            new_minimized_nodes.insert(&child.clone(), &child.clone());
        }
        new_curr_node = parent;
    }
    (new_curr_node, new_minimized_nodes)
}

#[pyfunction]
fn build_dawg(word_list: Vec<String>) -> Node {
    let root = Node::new();
    let mut minimized_nodes = HashMap::from([(&root, &root)]);
    let mut non_minimized_nodes: Vec<(Node, char, Node)> = Vec::new();
    let mut curr_node = root;
    let mut prev_word = String::new();
    for word in word_list.iter() {
        let pref_len = length_common_prefix(&prev_word, word);
        if !non_minimized_nodes.is_empty() {
            (curr_node, minimized_nodes) = minimize(
                curr_node.clone(),
                pref_len,
                minimized_nodes.clone(),
                non_minimized_nodes.clone(),
            );
        }
        for letter in word[pref_len as usize..].chars() {
            let next_node = Node::new();
            curr_node.children.insert(letter, next_node.clone());
            non_minimized_nodes.push((curr_node.clone(), letter, next_node.clone()));
            curr_node = &next_node;
        }
        curr_node.is_terminal = true;
        prev_word = word.to_string();
    }
    (&curr_node, minimized_nodes) = minimize(
        curr_node.clone(),
        0,
        minimized_nodes.clone(),
        non_minimized_nodes.clone(),
    );
    println!("{}", minimized_nodes.len());
    Node::new()
}

/*
fn build_dawg(lexicon: Vec<String>) -> Node {
    let root = Node::new();
    let mut minimized_nodes = HashMap::from([(root, root)]);
    let mut non_minimized_nodes: Vec<(&Node, char, Node)> = Vec::new();
    let mut curr_node = &mut root;
    let mut prev_word = String::new();
    for word in lexicon {
        let common_prefix_length = length_common_prefix(prev_word, word);

        if !non_minimized_nodes.is_empty() {
            curr_node = &mut minimize(
                &mut curr_node,
                common_prefix_length,
                &mut minimized_nodes,
                &mut non_minimized_nodes,
            );
        }

        for letter in word[common_prefix_length as usize..].chars() {
            let next_node = Node::new();
            curr_node.children[&letter] = next_node;
            non_minimized_nodes.push((&curr_node, letter, next_node));
            curr_node = &mut next_node;
        }

        curr_node.is_terminal = true;
        prev_word = word;
    }
    minimize(
        &mut curr_node,
        0,
        &mut minimized_nodes,
        &mut non_minimized_nodes,
    );
    println!("{}", minimized_nodes.len());
    root
} */
#[pyfunction]
fn file2dawg(path: &str) -> Node {
    let file = File::open(path).expect("Failed to open file");
    let reader = BufReader::new(file);
    let word_list: Vec<String> = reader.lines().filter_map(|line| line.ok()).collect();
    let dawg = build_dawg(word_list);
    dawg
}

#[pymodule]
fn dawgpyrust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Node>()?;
    m.add_function(wrap_pyfunction!(build_dawg, m)?)?;
    m.add_function(wrap_pyfunction!(length_common_prefix, m)?)?;
    m.add_function(wrap_pyfunction!(file2dawg, m)?)?;
    Ok(())
}
