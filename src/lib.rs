use numpy::ndarray::{Array1, ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, IxDyn, PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use std::collections::HashMap;
use std::f32::NAN;
use std::fs::File;
use std::io::{BufRead, BufReader};
use tqdm::Iter;

#[pyfunction]
fn fromstring<'py>(py: Python<'py>, s: &str, sep: &str) -> Bound<'py, PyArray1<f64>> {
    fn _fromstring(s: &str, sep: &str) -> Array1<f64> {
        s.split(sep)
            .map(|x| x.parse().unwrap())
            .collect::<Array1<f64>>()
    }

    let z = _fromstring(s, sep);
    z.into_pyarray_bound(py)
}

fn loglik_slice<F>(
    filename: &str,
    n_queries: usize,
    skip: usize,
    slice_fn: F,
) -> HashMap<String, Array1<f64>>
where
    F: Fn(&str) -> f64,
{
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);
    let mut doc2logliks = HashMap::new();
    for line in reader.lines().skip(skip).tqdm() {
        let line = line.unwrap();
        let mut parts = line.split("\t");
        let docid = parts.next().unwrap();
        let queryid: usize = parts.next().unwrap().parse().unwrap();

        if (queryid < n_queries) {
            let loglikstr = parts.next().unwrap();
            let loglik: f64 = slice_fn(loglikstr);
            doc2logliks
                .entry(docid.to_string())
                .or_insert(Array1::<f64>::from_elem([n_queries], NAN.into()))[queryid] = loglik;
        }
    }
    doc2logliks
}

#[pyfunction]
fn loglik_of_whole_query<'py>(
    py: Python<'py>,
    filename: &str,
    n_queries: usize,
    skip: usize,
) -> Bound<'py, PyDict> {
    fn slice_last(s: &str) -> f64 {
        let lastsep = s.rfind("|").unwrap();
        s[lastsep + 1..].parse().unwrap()
    }

    let doc2logliks = loglik_slice(filename, n_queries, skip, slice_last);

    let out = PyDict::new_bound(py);
    doc2logliks.into_iter().for_each(|(k, v)| {
        out.set_item(k, v.into_pyarray_bound(py)).unwrap();
    });
    out
}

#[pyfunction]
fn loglik_of_query_prefix<'py>(
    py: Python<'py>,
    filename: &str,
    n_queries: usize,
    skip: usize,
    prefix_length: usize,
) -> Bound<'py, PyDict> {
    let slice_nth = |s: &str| -> f64 {
        s.splitn(prefix_length + 1, "|")
            .nth(prefix_length - 1)
            .unwrap_or_else(|| {
                let lastsep = s.rfind("|").unwrap();
                &s[lastsep + 1..]
            })
            .parse()
            .expect(format!("Failed to parse string as a float: {}", s).as_str())
    };

    let doc2logliks = loglik_slice(filename, n_queries, skip, slice_nth);

    let out = PyDict::new_bound(py);
    doc2logliks.into_iter().for_each(|(k, v)| {
        out.set_item(k, v.into_pyarray_bound(py)).unwrap();
    });
    out
}

#[pymodule]
mod utils {

    #[pymodule_export]
    use super::fromstring;

    #[pymodule_export]
    use super::loglik_of_whole_query;

    #[pymodule_export]
    use super::loglik_of_query_prefix;
}
