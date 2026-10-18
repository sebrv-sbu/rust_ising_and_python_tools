mod ising;
mod ising_io;
mod anneal;
mod greedy;
use crate::ising_io::*;
use crate::ising::*;
use clap::{Parser,Subcommand,ArgAction};
use std::path::PathBuf;
use anyhow::{Result, ensure};

#[derive(Parser)]
#[command(about = "Implementation of High Temperature Simulated \
Annealing Ising Problem with Approximations for Convergence Rate")]
struct Args{
  input_file: PathBuf,
  #[arg(long = "ising_representation", value_enum, 
    default_value_t = IsingRepresentation::Disjoint)]
  ising_representation: IsingRepresentation,
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
  #[command(about = "Theoretical Convergence Rate, Low Temperature")]
  LowConv{
    #[arg(short = 'T', long = "temperature_override", default_value = None)]
    temperature_override:Option<f64>,
    #[arg(short = 'p', long = "percent", default_value = None)]
    percent_opt:Option<f64>,
    #[arg(short = 't', long = "approx_hitting_time",
      action = ArgAction::SetTrue)]
    compute_approx_hitting_time:bool,
    #[arg(short = 'r', long = "human_readable", action = ArgAction::SetTrue)]
    human_readable:bool,
  },
  #[command(about= "Asymptotic Matching for Convergence Rate")]
  AsymptoticMatch{
    #[arg(short = 'T', long = "temperature_override", default_value = None)]
    temperature_override:Option<f64>,
    #[arg(short = 'p', long = "percent", default_value = None)]
    percent_opt:Option<f64>,
    #[arg(short = 't', long = "approx_hitting_time",
      action = ArgAction::SetTrue)]
    compute_approx_hitting_time:bool,
    #[arg(short = 'r', long = "human_readable", action = ArgAction::SetTrue)]
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
    #[arg(short='l', long="append local minimum approx",
      action=ArgAction::SetTrue)]
    loc_min_approx:bool,
    #[arg(long="retries", default_value="5")]
    loc_min_retries:usize,
    #[arg(long="descending_retries", action=ArgAction::SetTrue)]
    loc_min_recursive:bool,
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

fn run_mode<I:Ising + std::fmt::Debug>(args: Args) -> Result<()>{
  let input_file = &args.input_file;
  let (mut ising_model, 
    temperature_from_file,
    ground_state_opt,
    local_state_opt) = from_ising_file::<I>( input_file )?;
  #[cfg(debug_assertions)]{
  println!("{:#?}", ising_model);   // pretty-printed with indentation
  }
  match args.mode{
    Mode::Inf{stationary_file_name, max_steps, runs, ground_state_provided} 
    => {
      infinite_temperature_option(
        &mut ising_model,
        max_steps,
        runs,
        stationary_file_name,
        ground_state_opt,
        ground_state_provided)?
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
        ground_state_provided)?
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
      let perturbation_result = high_conv_output(temperature, 
        human_readable, 
        ground_state_opt.as_ref(),
        &mut ising_model);
      if compute_approx_hitting_time{
        compute_hitting_time_output_high_t(human_readable, percent_opt,
          perturbation_result, &mut ising_model)?;
      }
    }
    Mode::GroundStateApprox{
      inner_sweeps,
      outer_sweeps,
      human_readable,
      loc_min_approx,
      modify_input_file,
      loc_min_retries,
      loc_min_recursive,
    } => {
      greedy_approx_output(inner_sweeps,
        outer_sweeps,
        human_readable,
        loc_min_approx,
        modify_input_file,
        loc_min_retries,
        loc_min_recursive,
        &mut ising_model,
        args.input_file)?;
    },

    Mode::GenFile{
      n:_, dim:_, sizes:_,
      all_ones:_, 
      edge_dist:_, field_dist:_, ext_dist:_, 
      seed_opt:_, temperature:_, file_seed_opt:_
    }
    => {
      eprintln!("What?");
    },
  Mode::LowConv{temperature_override,
  percent_opt, compute_approx_hitting_time, human_readable} => {
    let temperature = temperature_override.unwrap_or(temperature_from_file);
    let perturbation = low_conv_output(temperature, human_readable,
      local_state_opt.as_ref(), &mut ising_model)?;
    if compute_approx_hitting_time {
      compute_hitting_time_output_low_mid_t(human_readable,
        percent_opt, perturbation, &mut ising_model)?;
      }
    },
  Mode::AsymptoticMatch{temperature_override,
  percent_opt, compute_approx_hitting_time, human_readable} => {
  let temperature = temperature_override.unwrap_or(temperature_from_file);
  let perturbation = asymptotic_match_output(
    temperature,
    human_readable,
    local_state_opt.as_ref(),
    ground_state_opt.as_ref(),
    &mut ising_model
  )?;
  if compute_approx_hitting_time{
    compute_hitting_time_output_low_mid_t(human_readable, 
      percent_opt, perturbation, &mut ising_model)?;
      }
    },
  }
  Ok(())
}

fn main() -> Result<()>{
  let args = Args::parse();
  let input_file = &args.input_file;
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
      &input_file,
      ising_spec,
      temperature,
      file_seed,
      dim,
      sizes)?;
    return Ok(());
    }
  match args.ising_representation {
    IsingRepresentation::Disjoint => run_mode::<IsingDisjoint>(args),
    IsingRepresentation::Packed => run_mode::<IsingPacked>(args),
  }?;
  Ok(())
}
