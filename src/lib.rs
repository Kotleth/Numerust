use unicode_width::UnicodeWidthStr;

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
pub fn gauss_seidel(matrix: &[f32], b_slice: &[f32], num_rows: usize, num_columns: usize) -> (*const f32, usize){
    let a_matrix = make_matrix(matrix, num_rows, num_columns).unwrap();
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
    for _ in 0..12 {
        new_x_vect = mat_add(mat_mul(bsc_mat.clone(), new_x_vect).unwrap(), c_mat.clone()).unwrap();
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

pub fn tri_mat_inv(mat_1: Vec<Vec<f32>>, shape: i32) -> Result<Vec<Vec<f32>>, String> {
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
                    //     X[i,k] = -L[j, i:j-1]*X[i:j-1,j]/L[j,j]
                } else { return Err(String::from("Type of a triangular matrix has to be L or R")) }

            } else if i < j {
                if shape == 1 {
                    new_mat[i].push(0.0);
                } else if shape == 0 { // left-side triangular matrix TODO This need fixes
                    let mut temp_value = 0.0;
                    for k in j..i {
                        temp_value += mat_1[i][k] * new_mat[k][j];
                    }
                    new_mat[i].push(-temp_value/mat_1[i][i]);
                } else { return Err(String::from("Type of a triangular matrix has to be L or R")) }
            }
        }
    }
    Ok(new_mat)
}

pub fn mat_mul(mat_1: Vec<Vec<f32>>, mat_2: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>, String> {
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

pub fn mat_add(mat_1: Vec<Vec<f32>>, mat_2: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>, String> {
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

pub fn mat_neg(mat_1: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
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
    let mut file = File::create("XD.txt").unwrap();
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