use crate::ising::*;
use bitvec::prelude::*;
use anyhow::{bail, Context, Result};

const RECORD_EPS: f64 = 1e-9;
const RECOMPUTE_INTERVAL: usize = 1 << 20;

pub fn stationary_infinite_temperature_optimise<I:Ising>(
  ising: &mut I,
  max_steps:usize) -> (BitVec, f64, usize){
  let mut step = 0;
  (0..max_steps).fold(
    (ising.config_to_bitvec(), ising.cost(),0),
    |(min_conf, min_cost, hit), _|{
      ising.inf_anneal();
      step += 1;
      if ising.cost() < min_cost - RECORD_EPS{
        (ising.config_to_bitvec(), ising.cost(), step)
      } else {
        (min_conf, min_cost, hit)
      }
    })
}

pub fn stationary_infinite_temperature_hit<I:Ising>(
  ising: &mut I, 
  ground_state: &BitVec,
  max_steps:usize) -> Option<usize>{
  let mut hitting_time = 0;
  while hitting_time < max_steps{
    if ising.config_eq(ground_state){
      return Some(hitting_time)
    }
    ising.inf_anneal();
    hitting_time += 1;
  }
  if ising.config_eq(ground_state){
    return Some(hitting_time)
  }
  None
}

pub fn high_temp_theoretical_perturbation_naive<I:Ising>(
  ising: &mut I,
  ground_state_opt: Option<&BitVec>,
  temperature: f64
  )->(bool, f64){
  /* since $N/2^{2N+1}$ is quite a tiny number, *
   * we will to change the method.  */
  let mut sum = 0.0;
  let mut sign = false;
  for node in 0..ising.n_points(){
    let ground_state_cluster: Option<BitVec> = 
      ground_state_opt.map(|gs| {
        ising.neighbours(node).iter().map(|&nb| gs[nb]).collect()
      });
    let ground_state_center:Option<bool> = ground_state_opt.map(|gs| gs[node]);
    let mut node_sum = 0.0;
    for cluster_index in 0 .. 1<<ising.deg(){
      let cluster = &cluster_index.view_bits::<LocalBits>()[..ising.deg()]; 
      node_sum += ising.cluster_cost(
        temperature,
        node,
        cluster,
        ground_state_cluster.as_deref(),
        ground_state_center);
    }
    #[cfg(debug_assertions)]
    println!("node: {}: sum = {}", node, node_sum);
    sum += node_sum;
  }
  if sum < 0.0{
    sign = true;
    sum = -1.0 * sum;
  }
  (
    sign,
    sum.log2()-((ising.n_points() + 1+ising.deg()) as f64) +
      (ising.n_points() as f64).log2()
  )
}

pub fn stationary_finite_temperature_optimise<I:Ising>(
  ising: &mut I,
  max_steps:usize,
  temperature:f64
  )->(BitVec, f64, usize){
  let mut step = 0;
  (0..max_steps).fold(
    (ising.config_to_bitvec(), ising.cost(), 0),
    |(min_conf, min_cost, hit), _|{
      ising.anneal(temperature);
      if step % RECOMPUTE_INTERVAL == 0{
        ising.init_cost();
      }
      step += 1;
      if ising.cost() < min_cost - RECORD_EPS{
        (ising.config_to_bitvec(), ising.cost(), step)
      } else {
        (min_conf, min_cost, hit)
      }
    })
}

pub fn stationary_finite_temperature_hit<I:Ising>(
  ising: &mut I,
  ground_state: &BitVec,
  max_steps:usize,
  temperature:f64) -> Option<usize>
{
  let mut hitting_time = 0;
  while hitting_time < max_steps{
    if ising.config_eq(ground_state){
      return Some(hitting_time)
    }
    ising.anneal(temperature);
    hitting_time += 1;
  }
  if ising.config_eq(ground_state){
    return Some(hitting_time)
  }
  None
}

pub fn approx_num_steps_for_percent<I:Ising>(
  ising:&mut I,
  percent:f64,
  temperature_opt:Option<f64>,
  perturbation_opt:Option<f64>,
  local_min_approx_opt:Option<&BitVec>,
  ) -> Result<usize> {
  let perturbation = match perturbation_opt{
    Some(p) => { p },
    None => {
      match temperature_opt {
        None => {
          bail!("anneal.rs: neither perturbation nor temperature provided \
            to approx_num_steps. Cannot compute without at least one of \
            these being present.");
        },
        Some (temperature) => {
          let local_min_approx = local_min_approx_opt.context("cannot compute \
            low temperature perturbation for approx_num_steps_for_percent \
            without a local minimum provided")?;
          eprintln!("anneal.rs warning: using low temperature perturbation \
            to compute approximate number of steps needed to attain \
            perturbation.");
            ising.low_temp_pert(local_min_approx, temperature)
          }
        }
      }
    };
    if percent >= 1.0 || percent <= 0.0 {
      bail!(format!("anneal.rs: percentage {percent} outside (0,1)"));
    }
    if perturbation < 0.0 || perturbation > 1.0 {
      bail!(format!("anneal.rs: computer perturbation = {perturbation} \
          outside of range  [-1, 0] - perturbation may not be physically \
          meaningful"));
    }
    let ln_q = (-perturbation).ln_1p();
    let ln_fail_target = (-percent).ln_1p();
    let t = ln_fail_target / ln_q;
    Ok(t as usize)
}

pub fn approx_num_steps_for_percent_log_perturbation<I:Ising>(
  ising: &mut I,
  percent:f64,
  temperature_opt:Option<f64>,
  log_2_perturbation_and_sign_opt: Option<(bool, f64)>,
  ground_state_opt:Option<&BitVec>,
  ) -> Result<usize> {
  /* We will have to use some rust things to do this in *
   * a numerically stable way                           */
  let log_2_perturbation:f64;
  let sign:bool;
  if let Some(log_2_perturbation_and_sign) = log_2_perturbation_and_sign_opt{
    sign=log_2_perturbation_and_sign.0;
    log_2_perturbation = log_2_perturbation_and_sign.1;
    if ground_state_opt.is_some(){
    eprintln!("anneal.rs warning: Ground State provided to \
      approx_num_steps_for_percent is being ignored");
    } 
    if temperature_opt.is_some(){
      eprintln!("anneal.rs warning: Temperature \
      provided to approx_num_steps_for_percent is being ignored");
    }
  } else {
    match temperature_opt{
      None => {
      bail!("anneal.rs: neither log_2_perturbation nor \
        temperature was provided to \
        approx_num_steps_for_percent_log_perturbation. Cannot \
        compute without at least one of these being present.");
      }
      Some(temperature) => {
        println!("anneal.rs warning: approx_num_steps_for_percent \
          with no given perturbation uses the high temperature perturbation.");
      (sign, log_2_perturbation)= high_temp_theoretical_perturbation_naive(
      ising,
      ground_state_opt,
      temperature);
      }
    }
  }
   /* Now we will implement $1-(1-x)^t=p$ for p as the target percent. Here, *
    * we can solve and this gives $t=\ln(1-target)/\ln(1-x)$. We use log2    *
    * since we are given values in log base 2 anyway rather than ln.         *
    *                                                                        *
    * We borrow rust's ln_1p here which is optimized for logarithms of       *
    * numbers close to 1 and then solve.                                     */
  if percent >= 1.0 || percent <= 0.0{
    bail!(format!("anneal.rs error: percentage {percent} outside [0,1], \
    something must have gone wrong."));
  }
  let log2_1p = |y:f64| y.ln_1p() * std::f64::consts::LOG2_E;
  let x = log_2_perturbation.exp2()
    .copysign(if sign {-1.0} else {1.0}) - (-(ising.n_points() as f64)).exp2();
  
  let log2_q = log2_1p(x);
  let log2_fail_target = log2_1p(-percent);

  let t = log2_fail_target/log2_q;
  if x > 0.0 || x < -1.0 {
    bail!(format!("anneal.rs error: computed per-step rate x = {x} \
        outside [0,1] — perturbation may exceed base rate; result may not be \
        physically meaningful."));
  }
  Ok(t as usize)
}

pub fn asymptotic_matching<I:Ising>(
  ising: &mut I,
  local_min_approx:&BitVec,
  ground_state: &BitVec,
  temperature: f64,
  ) -> f64{
  let mut sum = 0.0;
  let high_temp_weight_func = |energy:f64, temp:f64| 
      (-energy.abs()/(temp * temp)).exp();
  let low_temp_weight_func = |energy:f64, temperature:f64| 
    1.0-high_temp_weight_func(energy, temperature);
  for node in 0..ising.n_points(){
    let ground_state_cluster:BitVec;
    let ground_state_cluster_opt: Option<&Bits>;
    let ground_state_center:Option<bool>;
    let neighbours = ising.neighbours(node);
      ground_state_cluster = (0..neighbours.len())
        .map(|i| ground_state[neighbours[i]])
        .collect();
      ground_state_cluster_opt = Some(ground_state_cluster.as_bitslice());
      ground_state_center = Some(ground_state[node]);
    let mut node_sum = 0.0;
      for cluster_index in 0 .. 1<<ising.deg(){
        let cluster = &cluster_index.view_bits::<LocalBits>()[..ising.deg()]; 
        node_sum += ising.cluster_cost_weighted(
          temperature,
          node,
          cluster,
          ground_state_cluster_opt,
          ground_state_center,
          high_temp_weight_func);
      }
      #[cfg(debug_assertions)]
      println!("node: {}: sum = {}", node, node_sum);
      sum += node_sum;
    }
  if sum < 0.0{
    sum = -1.0 * sum;
  }
  let low_temp_contrib = ising.low_temp_pert_weighted(local_min_approx,
    temperature,
    low_temp_weight_func);
  let high_temp_contrib = (sum*ising.n_points() as f64) / ((ising.n_points() + 
      1 + ising.deg()) as f64).exp2();
  1.0 + high_temp_contrib - low_temp_contrib
}
