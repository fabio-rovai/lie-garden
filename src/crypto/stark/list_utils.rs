use super::field::Zero;
use std::ops::Mul;

pub fn remove_trailing_elements<F: PartialEq + Clone>(
    list: &[F],
    element: &F,
) -> Vec<F> {
    if let Some(last) = list.last() {
        if last == element {
            return list
                .iter()
                .rev()
                .skip_while(|x| *x == element)
                .cloned()
                .collect::<Vec<F>>()
                .into_iter()
                .rev()
                .collect();
        }
    }
    list.to_vec()
}

pub fn two_lists_tuple_operation<F, T: Clone + Zero>(f: &[T], g: &[T], func: F) -> Vec<T>
where
    F: Fn(&T, &T) -> T,
{
    let max_len = f.len().max(g.len());
    let mut padded_f = f.to_vec();
    let mut padded_g = g.to_vec();

    padded_f.resize(max_len, T::zero());
    padded_g.resize(max_len, T::zero());

    padded_f
        .iter()
        .zip(padded_g.iter())
        .map(|(a, b)| func(a, b))
        .collect()
}

pub fn scalar_multiplication<T: Mul<Output = T> + Clone>(list: &[T], scalar: &T) -> Vec<T> {
    list.iter()
        .map(|elem| elem.clone() * scalar.clone())
        .collect()
}

pub fn prod<T>(values: &[T]) -> T
where
    T: Mul<Output = T> + Clone + Default,
{
    let len = values.len();
    if len == 0 {
        return T::default();
    }
    if len == 1 {
        return values[0].clone();
    }
    let mid = len / 2;
    prod(&values[..mid]) * prod(&values[mid..])
}
