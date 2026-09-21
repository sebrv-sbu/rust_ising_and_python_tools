mod ising;
mod ising_io;
mod anneal;
mod greedy;
use crate::ising_io::*;
use crate::anneal::*;
use crate::ising::*;
use crate::greedy::*;
use clap::{Parser,Subcommand,ArgAction};
use std::fs::File;
use std::path::PathBuf;
use std::io::{Write,BufWriter};
use bitvec::prelude::*;
use anyhow::{Result, ensure};

#[derive(Parser)]
#[command(about = "Implementation of High Temperature Simulated \
Annealing Ising Problem with Approximations for Convergence Rate")]
struct Args{
  input_file: PathBuf,
  #[command(subcommand)]
  mode: Mode,
}



#[derive(Subcommand)]
enum Mode{
  #[command(about = "Infinite Temperature Experiments")]
  Inf{
    #[arg(short = 's', long = "stationary_file",
      default_value = "stationary_hits_infinite.txt")]
    stationary_file_name:PathBuf,
    #[arg(short = 'n', long = "max_steps", default_value="100")]
    max_steps:usize,
    #[arg(short = 'r', long = "runs", default_value = "100")]
    runs:usize,
    #[arg(short = 'g', long = "ground_state_provided", 
      action = ArgAction::SetTrue)]
    ground_state_provided:bool,
  },
  #[command(about = "Theoretical Convergence Rate, High Temperature")]
  HighConv{
    #[arg(short = 'T', long = "temperature_override", default_value = None)]
    temperature_override:Option<f64>,
    #[arg(short = 'p', long = "percent", default_value = None)]
    percent_opt:Option<f64>,
    #[arg(short = 't', long = "approx_hitting_time",
      action = ArgAction::SetTrue)]
    compute_approx_hitting_time:bool,
    #[arg(short = 'r', long ="human_readable", action = ArgAction::SetTrue)]
    human_readable:bool,
  },
  #[command(about = "Finite Temperature Experiments")]
  FiniteT{
    #[arg(short = 's', long = "stationary_file",
      default_value = "stationary_hits_finite.txt")]
    stationary_file_name:PathBuf,
    #[arg(short = 'n', long = "max_steps", default_value="100")]
    max_steps:usize,
    #[arg(short = 'r', long = "runs", default_value = "100")]
    runs:usize,
    #[arg(short = 'T', long = "temperature_override", default_value=None)]
    temperature_override: Option<f64>,
    #[arg(short = 'g', long = "hitting_time_with_ground_state", 
      action = ArgAction::SetTrue)]
    ground_state_provided:bool,
  },
  #[command(about="Approximate Ground State with Greedy Algorithm")]
  GroundStateApprox{
    #[arg(short='o', long="outer_sweeps", default_value="10")]
    outer_sweeps:usize,
    #[arg(short='i', long="inner_sweeps", default_value="10")]
    inner_sweeps:usize,
    #[arg(short='r', long="human_readable", action=ArgAction::SetTrue)]
    human_readable:bool,
    #[arg(short='w', long="modify_input_file", action=ArgAction::SetTrue)]
    modify_input_file:bool,
  },
  #[command(about="Generate Ising Files according to Specifications")]
  GenFile{ 
    n:usize,
    dim:usize,
    sizes:Vec<usize>,
    #[arg(short = 'o', long="all_ones", action=ArgAction::SetTrue)]
    all_ones:bool,
    #[arg(short = 'e', long="edge_dist", default_value="normal:0.0,1.0")]
    edge_dist:Dist,
    #[arg(short = 'f', long="magnetic_field", default_value="normal:0.0,1.0")]
    field_dist:Dist,
    #[arg(short = 'm', long="external_field", default_value="normal:0.0,1.0")]
    ext_dist:Dist,
    #[arg(short = 's', long = "write_seed", default_value=None)]
    seed_opt:Option<u64>,
    #[arg(short = 'T', long = "temperature", default_value = "1.0")]
    temperature:f64,
    #[arg(long = "file_seed", default_value=None)]
    file_seed_opt:Option<u64>,
  }
}

fn finite_temperature_option<I:Ising>(
  ising_model: &mut I,
  max_steps: usize,
  runs: usize,
  stationary_file_name: PathBuf,
  temperature:f64,
  ground_state_opt:Option<BitVec>,
  ground_state_provided:bool,
  ){
  let stationary_file = File::create(stationary_file_name).expect("main.rs\
    error:could not create new stationary file");
  let mut stationary_out = BufWriter::new(stationary_file);
  if ground_state_provided{
    if ground_state_opt.is_none(){
      eprintln!("main.rs error: no ground state was provided. Running \
        simulations with all 0s as ground state.");
      return;
    }
    let ground_state = ground_state_opt.unwrap();
    for _ in 0..runs{
    let hitting_time = stationary_finite_temperature_hit(
      ising_model,
      &ground_state,
      max_steps,
      temperature
    );
      match hitting_time{
        Some(steps) => writeln!(stationary_out, "{}", steps).unwrap(),
        None => writeln!(stationary_out, "NA").unwrap()
      }
      ising_model.init_spins();
      ising_model.init_cost();
    } 
  } else {
    writeln!(stationary_out, "{}\t{}\t{}\t{}", "best_configuration",
      "best_cost", "best_cost_fpb", "hitting_time").unwrap();
    for _ in 0..runs{
      let (best_conf, best_cost, hit) 
        = stationary_finite_temperature_optimise(
          ising_model,
          max_steps,
          temperature);
      let best_conf_string = bitvec_to_hex_string(&best_conf);
      writeln!(stationary_out, "{}\t{:.6e}\t{:016x}\t{}", best_conf_string,
        best_cost, best_cost.to_bits(), hit).unwrap();
    }
  }
}

fn infinite_temperature_option<I:Ising>(
  ising_model: &mut I,
  max_steps:usize,
  runs:usize,
  stationary_file_name:PathBuf,
  ground_state_opt:Option<BitVec>,
  ground_state_provided:bool
  ){
  let stationary_file = File::create(stationary_file_name).expect("main.rs\
Error:could not create new stationary file");
  let mut stationary_out = BufWriter::new(stationary_file);
  if ground_state_provided{
    if ground_state_opt.is_none(){
      eprintln!("main.rs warning: no ground state was provided. Running \
        simulations with all 0s as ground state.");
      return;
    }
    let ground_state = ground_state_opt.unwrap();
    for _ in 0..runs{
      let hitting_time = stationary_infinite_temperature_hit(
        ising_model,
        &ground_state,
        max_steps
        );
      match hitting_time{
        Some(steps) => writeln!(stationary_out, "{}", steps).unwrap(),
        None => writeln!(stationary_out, "NA").unwrap()
      }
      ising_model.init_spins();
      ising_model.init_cost();
    }
  } else {
    writeln!(stationary_out, "{}\t{}\t{}\t{}", "best_configuration",
      "best_cost", "best_cost_fpb", "hitting_time").unwrap();
    for _ in 0..runs{
      let (best_conf, best_cost, hit)
        = stationary_infinite_temperature_optimise(
          ising_model,
          max_steps);
      let best_conf_string = bitvec_to_hex_string(&best_conf);
      writeln!(stationary_out, "{}\t{:.6e}\t{:016x}\t{}", best_conf_string,
        best_cost, best_cost.to_bits(), hit).unwrap();
    }
  }
}

fn main() -> Result<()>{
  let args = Args::parse();
  let input_file = args.input_file;

  if let Mode::GenFile{
      n, dim, sizes,
      all_ones, 
      edge_dist, field_dist, ext_dist, 
      seed_opt, temperature, file_seed_opt 
  } = args.mode
  {
    if all_ones{
      eprintln!("Warning: -o, --all_ones will ignore all distribution flags.");
    }
    let ising: IsingModels = if all_ones{ 
     IsingModels::AllDownAllOnesGroundState 
    } else { IsingModels::EA{edge: edge_dist, mag:field_dist, ext:ext_dist} };
    ensure!(sizes.len() == dim, "dimension and sizes must have the same length"
      );
    ensure!(
      *(&sizes.iter().product::<usize>()) == n,
      "product of sizes ({}) must equal the number of nodes ({n})",
      sizes.iter().product::<usize>(),
    );
    let sizes: Box<[usize]> = sizes.into_boxed_slice();
    let seed = seed_opt.unwrap_or_else(rand::random);
    let ising_spec = generate_ising_file_specifiers(
    ising,
    dim,
    n,
    seed)?;
    let file_seed = file_seed_opt.unwrap_or_else(rand::random);
    generate_ising_file(
      input_file,
      ising_spec,
      temperature,
      file_seed,
      dim,
      sizes)?;
    return Ok(());
    }


  let (mut ising_model, temperature_from_file, ground_state_opt) =
    from_ising_file_disjoint_simple(
    &input_file
  );
   
  match args.mode{
    Mode::Inf{stationary_file_name, max_steps, runs, ground_state_provided} 
    => {
      infinite_temperature_option(
        &mut ising_model,
        max_steps,
        runs,
        stationary_file_name,
        ground_state_opt,
        ground_state_provided)
      }
    Mode::FiniteT{
      stationary_file_name,
      max_steps,
      runs,
      temperature_override,
      ground_state_provided} => {
      let temperature:f64;
      match temperature_override{
        Some(temperature_opt) => { 
          temperature = temperature_opt;
        }
        None => {
          temperature = temperature_from_file;
        }
      }
      finite_temperature_option(
        &mut ising_model,
        max_steps,
        runs,
        stationary_file_name,
        temperature,
        ground_state_opt,
        ground_state_provided);
    }
    Mode::HighConv{
      temperature_override,
      percent_opt,
      compute_approx_hitting_time,
      human_readable,
    } => {
      let temperature:f64;
      match temperature_override{
        Some(temperature_opt) => { 
          temperature = temperature_opt;
        }
        None => {
          temperature = temperature_from_file;
        }
      }
      let (sign, log_sum) = theoretical_perturbation_naive(
        &mut ising_model,
        &ground_state_opt,
        temperature);
      if human_readable{
        println!("log2 absolute perturbation:{}", log_sum);
        println!("sign:{}", if sign {"-"} else {"+"});
      } else {
        println!("{}, {}", log_sum, if sign {"-"} else {"+"});
      }
      if compute_approx_hitting_time{
        match approx_num_steps_for_percent(
          &mut ising_model,
          percent_opt,
          None,
          Some((log_sum, sign)),
          &None) {
          Ok(steps) => {
            let percent=percent_opt.unwrap_or(0.25);
            if human_readable {
              println!("Takes ~ {steps} steps to have {percent} \
              probability of hitting")
            } else {
              println!("{}, {}", steps, percent);
            }
          },
          Err(e) => eprintln!("{e}"),
        }
      }
    }
    Mode::GroundStateApprox{
      inner_sweeps,
      outer_sweeps,
      human_readable,
      modify_input_file
    } => {
      let ground_state_approx=greedy(
        &mut ising_model,
        inner_sweeps,
        outer_sweeps);
      if human_readable && modify_input_file {
        eprintln!("main.rs error: human_readable and modify_input_files \
          cannot both be set.");
        std::process::exit(1);
      }
      if human_readable {
        println!("Greedy algorithm found the following configuration:");
        println!("{}",
          ground_state_approx.iter()
            .by_vals()
            .map(|spin| if !spin {"-1"} else {"+1"})
            .collect::<Vec<_>>()
            .join(" ")
        );
        return Ok(());
      }
      let conf_len = ground_state_approx.len();
      let config_str: String = ground_state_approx
        .chunks(64)
        .map(|chunk| { chunk.iter()
          .by_vals()
          .enumerate()
          .fold(0u64, |n, (i, bit)| n | ((bit as u64) << i))
        })
        .flat_map(|num| num.to_le_bytes())
        .take((conf_len + 7) / 8)
        .map(|byte| format!("{:02x}", byte))
        .collect::<Vec<String>>()
        .join(" ");

      if !modify_input_file{
        println!("{}", config_str);
       }
      if modify_input_file{
        write_ground_state(config_str, input_file)?;
      }
    },
    Mode::GenFile{
      n, dim, sizes,
      all_ones, 
      edge_dist, field_dist, ext_dist, 
      seed_opt, temperature, file_seed_opt
    }
    => {
      eprintln!("What?");
    },
  }
  Ok(())
}


