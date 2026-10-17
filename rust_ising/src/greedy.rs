use crate::ising::*;
use bitvec::prelude::*;
pub fn greedy<I:Ising>(
  ising:&mut I, 
  inner_sweeps:usize, 
  outer_sweeps:usize
  ) -> (BitVec, f64)
{
  let mut results = Vec::new();
  for _ in 0..outer_sweeps{
    ising.init_spins();
    ising.init_cost();
    let mut new_cost = ising.cost();
    for _ in 0..inner_sweeps{
      let mut changed=false;
      
      for node in 0..ising.n_points() {
        let delta = ising.cost_diff(node);
        if delta < 0.0 {
          ising.flip_spin(node, delta);
          changed=true;
        }
      }
      new_cost = ising.cost();
      if !changed{
        break
      }
    }
    results.push((ising.config_to_bitvec(), new_cost));
  }
  results.into_iter()
    .min_by(|(_,cost_a), (_, cost_b)| cost_a.total_cmp(cost_b))
    .unwrap()
}

pub fn local_global_min_approx<I:Ising>(
  ising: &mut I,
  outer_sweeps:usize,
  inner_sweeps:usize,
  loc_min_retries:usize,
  loc_min_recursive:bool) -> (BitVec, BitVec){
  let (first_state, first_cost) = 
    greedy(ising, inner_sweeps, outer_sweeps);
  let mut second_state = first_state.clone();
  let mut second_cost = first_cost;
  let mut retries = 0;
  let mut inner_sweeps_loc = inner_sweeps;
  let mut outer_sweeps_loc = outer_sweeps;
  while inner_sweeps_loc > 0 && outer_sweeps_loc > 0{
    while second_state == first_state && retries < loc_min_retries{
      (second_state, second_cost) = greedy(
        ising, inner_sweeps_loc, outer_sweeps_loc
      );
      retries += 1;
    }
    if !loc_min_recursive{
      outer_sweeps_loc = 0;
    } else {
      inner_sweeps_loc -= 1;
      outer_sweeps_loc -=1;
    }
  }
    if second_cost < first_cost {
      (second_state, first_state)
    } else {
      (first_state, second_state)
    }
}
