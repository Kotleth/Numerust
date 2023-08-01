use unicode_width::UnicodeWidthStr;
use std::time::Instant;
use std::{time};
use std::thread;
use std::sync::mpsc;

// pub fn matrix_inv(x_arr: &[f32], n: usize) -> (*const f32, usize) {
//     decomposition_lu
// }
#[no_mangle]
pub fn matrix_inv(x_arr: &[f32], n: usize) -> (*const f32, usize) {
    let mut l: Vec<Vec<f32>> = Vec::new();
    let mut u: Vec<Vec<f32>> = Vec::new();
    for i in 0..n {
        l.push(Vec::new());
        u.push(Vec::new());
        for j in 0..n {
            if i <= j {
                u[i].push(x_arr[i * n + j]);
                for k in 0..i {
                    u[i][j] -= l[i][k] * u[k][j];
                }
                if i == j {
                    l[i].push(1.0);
                } else {
                    l[i].push(0.0);
                }
            } else {
                l[i].push(x_arr[i * n + j]);
                for k in 0..j {
                    l[i][j] -= l[i][k] * u[k][j];
                }
                l[i][j] /= u[j][j];
                u[i].push(0.0);
            }
        }
    }
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let inv_l = tri_mat_inv(l.clone(), 1).unwrap();
        tx.send(inv_l).unwrap();
    });
    let inv_u = tri_mat_inv(u.clone(), 0).unwrap();
    // let inv_l = tri_mat_inv(l.clone(), 1).unwrap();

    let inv_l= rx.recv().unwrap();
    // handle.join().unwrap();
    let inv_mat = mat_mul(inv_u, inv_l).unwrap();
    let mut temp_vec = Vec::new();
    for i in inv_mat {
        for j in i {
            temp_vec.push(j);
        }
    }

    let slice_ptr = temp_vec.as_slice().as_ptr();
    let slice_len = temp_vec.len();
    std::mem::forget(temp_vec);
    // slice_ptr
    (slice_ptr, slice_len)
}

#[no_mangle]
pub fn least_square_approximation(x_arr: &[f32], y_arr: &[f32], degree: usize) -> (*const f32, usize)
{ // TODO check at the beginning if x_arr.len() == y_arr.len()
    let mut x_mat: Vec<Vec<f32>> = Vec::new();
    let mut y_vec: Vec<Vec<f32>> = Vec::new();
    for i in 0..degree {
        x_mat.push(Vec::new());
        y_vec.push(Vec::new());
        let mut y_sum = 0.0;
        for num in 0..y_arr.len() {
            y_sum += f32::powf(x_arr[num], i as f32) * y_arr[num];
        }
        y_vec[i].push(y_sum);
        for j in 0..degree {
            let mut x_sum = 0.0;
            for x in x_arr {
                x_sum += f32::powf(*x, i as f32 + j as f32);
            }
            x_mat[i].push(x_sum);
        }

    }
    // vis_mat(x_mat.clone());
    // vis_mat(y_vec.clone());
    let inverted_mat = inside_mat_inv(x_mat).unwrap();
    let final_vec = mat_mul(inverted_mat, y_vec).unwrap();
    // vis_mat(inverted_mat);
    let mut temp_vec = Vec::new();
    for i in &final_vec {
        temp_vec.push(i[0]);
    }
    let slice_ptr = temp_vec.as_slice().as_ptr();
    let slice_len = temp_vec.len();
    std::mem::forget(temp_vec);
    // slice_ptr
    (slice_ptr, slice_len)
    // we need to perform y_vec * x_mat^-1 now and it will be finished
}

#[no_mangle]
pub fn newton_optimisation_polynomial(x_multipliers: &[f32], x_first: f32, error: f32) -> f32 {
    let mut killswitch = 500;
    let mut x_zero = x_first;
    while killswitch > 1 {
        killswitch -= 1;
        x_zero = x_zero - polynomial_f(&x_multipliers, x_zero)/diff_quotient_polynomial(&x_multipliers, x_zero, x_zero/10.0);
        if polynomial_f(&x_multipliers, x_zero) > -error && polynomial_f(&x_multipliers, x_zero) < error {
            killswitch = 1;
        }
    }
    x_zero
}

#[no_mangle]
// pub fn gauss_seidel(matrix: &[f32], b_slice: &[f32], num_rows: usize, num_columns: usize) -> *const f32 {
pub fn gauss_seidel(matrix: &[f32], b_slice: &[f32], num_rows: usize, timeout: u64) -> (*const f32, usize){
    let a_matrix = make_matrix(matrix, num_rows, num_rows).unwrap();
    let mut b_vector: Vec<Vec<f32>> = Vec::new();
    let mut n = 0;
    for row in b_slice {
        b_vector.push(Vec::new());
        b_vector[n].push(*row);
        n += 1;
    }
    let mut x_vector: Vec<Vec<f32>> = Vec::new();
    for i in 0..b_vector.len() {
        x_vector.push(Vec::new());
        x_vector[i].push(1.0);
    }
    let mut i: usize = 0;
    let mut mat_up: Vec<Vec<f32>> = Vec::new();
    let mut mat_diag: Vec<Vec<f32>> = Vec::new();
    let mut mat_low: Vec<Vec<f32>> = Vec::new();
    for row in &a_matrix {
        let mut j: usize = 0;
        mat_up.push(Vec::new());
        mat_diag.push(Vec::new());
        mat_low.push(Vec::new());
        for element in row {
            if i < j {
                mat_up[i].push(*element);
                mat_diag[i].push(0.0);
                mat_low[i].push(0.0);
            } else if i == j {
                mat_up[i].push(0.0);
                mat_diag[i].push(*element);
                mat_low[i].push(0.0);
            } else if i > j {
                mat_up[i].push(0.0);
                mat_diag[i].push(0.0);
                mat_low[i].push(*element);
            }
            j += 1;
        }
        i += 1;
    }
    let mut new_x_vect = x_vector;
    let mat_ld_inv = tri_mat_inv(mat_add(mat_low.clone(), mat_diag.clone()).unwrap(), 1).unwrap();
    let bsc_mat = mat_mul(mat_neg(mat_ld_inv.clone()), mat_up.clone()).unwrap();
    let c_mat = mat_mul(mat_ld_inv.clone(), b_vector.clone()).unwrap();
    let start = Instant::now();
    let mut elapsed = start.elapsed();
    // let mut progress =
    while elapsed <= time::Duration::from_millis(timeout) {
        elapsed = start.elapsed();
        new_x_vect = mat_add(mat_mul(bsc_mat.clone(), new_x_vect).unwrap(), c_mat.clone()).unwrap();
        // let previous_x_vect = new_x_vect.copy();
    }
    let mut temp_vec = Vec::new();
    for i in &new_x_vect {
        temp_vec.push(i[0]);
    }
    let slice_ptr = temp_vec.as_slice().as_ptr();
    let slice_len = temp_vec.len();
    std::mem::forget(temp_vec);
    // slice_ptr
    (slice_ptr, slice_len)
}

fn tri_mat_inv(mat_1: Vec<Vec<f32>>, shape: i32) -> Result<Vec<Vec<f32>>, String> {
    // shape 0 means right-side triangular matrix and 1 is left-side triangular matrix.
    let mut new_mat = Vec::new();
    for i in 0..mat_1.len() {
        new_mat.push(Vec::new());
        for j in 0..mat_1[i].len() {
            if i == j {
                new_mat[i].push(1.0/mat_1[i][j]);
            } else if i > j {
                if shape == 0 { // right-side triangular matrix
                    new_mat[i].push(0.0);
                } else if shape == 1 { // left-side triangular matrix
                    let mut temp_value = 0.0;
                    for k in j..i {
                        temp_value += mat_1[i][k] * new_mat[k][j];
                    }
                    new_mat[i].push(-temp_value/mat_1[i][i]);
                } else { return Err(String::from("Type of a triangular matrix has to be L or R")) }

            } else if i < j {
                if shape == 1 {
                    new_mat[i].push(0.0);
                } else if shape == 0 {
                    let mut temp_value = 0.0;
                    for k in i..j {
                        temp_value += mat_1[k][j] * new_mat[i][k];
                    }
                    new_mat[i].push(-temp_value/mat_1[j][j]);
                } else { return Err(String::from("Type of a triangular matrix has to be L or R")) }
            }
        }
    }
    Ok(new_mat)
}

fn mat_mul(mat_1: Vec<Vec<f32>>, mat_2: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>, String> {
    let mut res_mat: Vec<Vec<f32>> = Vec::new();
    let x_len = mat_2[0].len();
    let y_len = mat_1.len();
    if  mat_2.len() != mat_1[0].len() {
        return Err(String::from("Matrix A number of rows and Matrix B number of columns must match."))
    };

    for i in 0..y_len {
        res_mat.push(Vec::new());
        for j in 0..x_len {
            let mut temp_value: f32 = 0.0;
            for k in 0..mat_2.len() {
                temp_value += mat_1[i][k] * mat_2[k][j];
            }
            res_mat[i].push(temp_value);
        }
    }
    Ok(res_mat)
}

fn mat_add(mat_1: Vec<Vec<f32>>, mat_2: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>, String> {
    let mut res_mat: Vec<Vec<f32>> = Vec::new();
    let x_len = mat_1.len();
    let y_len = mat_1[0].len();

    for row in &mat_1 {
        if row.len() != mat_1[0].len() {
            return Err(String::from("Matrix rows need to be the same length."));
        }
    }

    for row in &mat_2 {
        if row.len() != mat_2[0].len() {
            return Err(String::from("Matrix rows need to be the same length."));
        }
    }

    for i in 0..x_len {
        res_mat.push(Vec::new());
        for j in 0..y_len {
            res_mat[i].push(mat_1[i][j] + mat_2[i][j]);
        }
    }
    Ok(res_mat)
}

fn mat_neg(mat_1: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut neg_mat = Vec::new();
    let mut i = 0;
    for row in mat_1 {
        neg_mat.push(Vec::new());
        for value in row {
            neg_mat[i].push(-value);
        }
        i += 1;
    }
    neg_mat
}



fn diff_quotient_polynomial(x_multipliers: &[f32], x: f32, step: f32) -> f32{
    (polynomial_f(&x_multipliers, x + step) - polynomial_f(&x_multipliers, x - step))/(2.0 * step)
}

fn polynomial_f(x_multipliers: &[f32], x: f32) -> f32 {
    let mut i = x_multipliers.len();
    let mut func_result: f32 = 0.0;
    for multiplier in x_multipliers {
        i -= 1;
        func_result += f32::powf(x, i as f32) * multiplier;
    }
    func_result
}

fn decomposition_lu(x_arr: &[f32], n: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut l: Vec<Vec<f32>> = Vec::new();
    let mut u: Vec<Vec<f32>> = Vec::new();
    for i in 0..n {
        l.push(Vec::new());
        u.push(Vec::new());
        for j in 0..n {
            if i <= j {
                u[i].push(x_arr[i * n + j]);
                for k in 0..i {
                    u[i][j] -= l[i][k] * u[k][j];
                }
                if i == j {
                    l[i].push(1.0);
                } else {
                    l[i].push(0.0);
                }
            } else {
                l[i].push(x_arr[i * n + j]);
                for k in 0..j {
                    l[i][j] -= l[i][k] * u[k][j];
                }
                l[i][j] /= u[j][j];
                u[i].push(0.0);
            }
        }
    }
    (l, u)
}


use std::fs::File;
use std::io::prelude::*;


fn make_matrix(vector: &[f32], num_rows: usize, num_columns: usize) -> Result<Vec<Vec<f32>>, String> {
    let mut res_mat: Vec<Vec<f32>> = Vec::new();
    if vector.len()%num_rows != 0 {
            return Err(String::from("Matrix rows does not match data length."));
    } else if (vector.len()/num_rows)%num_columns != 0 {
            return Err(String::from("Matrix columns does not match data length."));
    } else {
        for i in 0..num_rows {
            res_mat.push(Vec::new());
            for j in 0..num_columns {
                res_mat[i].push(vector[i * num_rows + j]);
            }
        }
        Ok(res_mat)
    }
}

#[no_mangle]
pub fn vis_mat(vector: &[f32], num_rows: usize, num_columns: usize) -> std::io::Result<()> {
    let mut a_matrix = make_matrix(vector, num_rows, num_columns).unwrap();
    let mut width: usize = 0;
    let mut file = File::create("Matrix.txt").unwrap();
    for row in a_matrix.iter_mut() {
        for float in row.iter_mut() {
            if float.to_string().len() > width.to_string().len() {
                width = float.to_string().len();
            }
        }
    }

    for row in a_matrix.iter_mut() {
        file.write_all(b"| ").unwrap();
        for element in row.iter_mut() {
            let xd = format_width(&element.to_string(), width, ' ');
            file.write_all(xd.as_bytes()).unwrap();
        }
        file.write_all(b" |\n").unwrap();
    }
    file.flush().unwrap();
    Ok(())
}

fn inside_mat_inv(x_arr: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>, String> {
    let mut l: Vec<Vec<f32>> = Vec::new();
    let mut u: Vec<Vec<f32>> = Vec::new();
    let n = x_arr.len();
    for i in 0..n {
        l.push(Vec::new());
        u.push(Vec::new());
        for j in 0..n {
            if i <= j {
                u[i].push(x_arr[i][j]);
                for k in 0..i {
                    u[i][j] -= l[i][k] * u[k][j];
                }
                if i == j {
                    l[i].push(1.0);
                } else {
                    l[i].push(0.0);
                }
            } else {
                l[i].push(x_arr[i][j]);
                for k in 0..j {
                    l[i][j] -= l[i][k] * u[k][j];
                }
                l[i][j] /= u[j][j];
                u[i].push(0.0);
            }
        }
    }
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let inv_l = tri_mat_inv(l.clone(), 1).unwrap();
        tx.send(inv_l).unwrap();
    });
    let inv_u = tri_mat_inv(u.clone(), 0).unwrap();
    // let inv_l = tri_mat_inv(l.clone(), 1).unwrap();

    let inv_l= rx.recv().unwrap();
    // handle.join().unwrap();
    let inv_mat = mat_mul(inv_u, inv_l).unwrap();
    Ok(inv_mat)
}

fn format_width(input: &str, width: usize, fill_char: char) -> String {
    let input_width = UnicodeWidthStr::width(input);

    if input_width >= width {
        String::from(input)
    } else {
        let fill_count = width - input_width;
        let fill_string: String = std::iter::repeat(fill_char).take(fill_count).collect();
        format!("{}{}", input, fill_string)
    }
}