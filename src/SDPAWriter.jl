# Copyright (c) 2015: AmplNLWriter.jl contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# to do: fix documentation

import MathOptInterface

"""
    AbstractSolverCommand
An abstract type that allows over-riding the call behavior of the solver.
See also: [`call_solver`](@ref).
"""
abstract type AbstractSolverCommand end

"""
    call_solver(
        solver::AbstractSolverCommand,
        sdpa_filename::String,
        options::Vector{String},
        stdin::IO,
        stdout::IO,
    )::String
Execute the `solver` given the SDPA file at `sdpa_filename`, a vector of `options`,
and `stdin` and `stdout`. Return the filename of the resulting `.sol` file.
You can assume `sdpa_filename` ends in `.dat-s`, and that you can write a `.dat`
file to `replace(sdpa_filename, "model.dat-s" => "model.dat")`.
If anything goes wrong, throw a descriptive error.
"""
function call_solver end

# need to rework with binary call for SDPA solver
function call_solver(
    solver,
    sdpa_filename::String,
    options::Vector{String},
    stdin::IO,
    stdout::IO,
)

    read_path = full_output_path
    if m.use_WSL
        full_input_path = WSLize_path(full_input_path)
        full_output_path = WSLize_path(full_output_path)
        if m.verbosity != SILENT
            @info "Redirecting to WSL environment."
        end
    end
    if m.params == UNSTABLE_BUT_FAST
        arg = `-ds $full_input_path -o $full_output_path -pt 1`
    elseif m.params == STABLE_BUT_SLOW
        arg = `-ds $full_input_path -o $full_output_path -pt 2`
    else
        params_path = get_params_path(m)
        arg = `-ds $full_input_path -o $full_output_path -p $(params_path)`
    end

    if m.variant == :sdpa
        error_messages, miss = SDPA_jll.sdpa() do path
            run_binary(`$path $arg`, m.verbosity)
        end
    elseif m.use_WSL
        wsl_binary_path = dirname(normpath(m.binary_path))
        error_messages, miss = cd(wsl_binary_path) do
            var = string(m.variant)
            run_binary(`wsl ./$var $arg`, m.verbosity)
        end
    else
        error_messages, miss = withenv([prefix]) do
            run_binary(`$(m.binary_path) $arg`, m.verbosity)
        end
    end

    error_log_path = joinpath(m.tempdir, "errors.log")
    open(error_log_path, "w") do io
        print(io, error_messages)
    end

    if m.verbosity != SILENT
        if m.verbosity == VERBOSE && error_messages != ""
            println("error log: $error_log_path")
        end

        if miss
            @warn("'cholesky miss condition' warning detected; results may be unreliable. Try `presolve=true`, or see troubleshooting guide.")
        end
    end

    #read_results!(m, read_path, redundant_entries);

#=
    solver.f() do solver_path
        ret = run(
            pipeline(
                `$(solver_path) $(sdpa_filename) -AMPL $(options)`,
                stdin = stdin,
                stdout = stdout,
            ),
        )
        if ret.exitcode != 0
            error("Nonzero exit code: $(ret.exitcode)")
        end
    end
    return replace(sdpa_filename, "model.nl" => "model.sol")
=#
end

"""
    _solver_command(x::Union{Function,String})
Functionify the solver command so it can be called as follows:
```julia
foo = _solver_command(x)
foo() do path
    run(`\$(path) args...`)
end
```
"""
_solver_command(x::String) = _DefaultSolverCommand(f -> f(x))
_solver_command(x::Function) = _DefaultSolverCommand(x)
_solver_command(x::AbstractSolverCommand) = x

mutable struct Optimizer <: MOI.AbstractOptimizer
    inner::MOI.FileFormats.SDPA.Model
    solver_command::AbstractSolverCommand
    options::Dict{String,Any}
    stdin::Any
    stdout::Any
    results::MOI.FileFormats.NL.SolFileResults
    solve_time::Float64
end

"""
    Optimizer(
        solver_command::Union{String,Function},
        solver_args::Vector{String};
        stdin::Any = stdin,
        stdout:Any = stdout,
    )
Create a new Optimizer object.
## Arguments
 * `solver_command`: one of two things:
   * A `String` of the full path of an SDPA-compatible executable
   * A function that takes takes a function as input, initializes any
     environment as needed, calls the input function with a path to the
     initialized executable, and then destructs the environment.
 * `solver_args`: a vector of `String` arguments passed solver executable.
   However, prefer passing `key=value` options via `MOI.RawOptimizerAttribute`.
 * `stdin` and `stdio`: arguments passed to `Base.pipeline` to redirect IO. See
   the Julia documentation for more details by typing `? pipeline` at the Julia
   REPL.
## Examples
A string to an executable:
```julia
Optimizer("/path/to/ipopt.exe")
```
A function or string provided by a package:
```julia
Optimizer(Ipopt.amplexe)
# or
Optimizer(Ipopt_jll.amplexe)
```
A custom function
```julia
function solver_command(f::Function)
    # Create environment ...
    ret = f("/path/to/ipopt")
    # Destruct environment ...
    return ret
end
Optimizer(solver_command)
```
The following two calls are equivalent:
```julia
# Okay:
model = Optimizer(Ipopt_jll.amplexe, ["print_level=0"])
# Better:
model = Optimizer(Ipopt_jll.amplexe)
MOI.set(model, MOI.RawOptimizerAttribute("print_level"), 0
```
"""
function Optimizer(
    solver_command::Union{AbstractSolverCommand,String,Function} = "",
    solver_args::Vector{String} = String[];
    stdin::Any = stdin,
    stdout::Any = stdout
)
    return Optimizer(
        MOI.FileFormats.SDPA.Model(),
        _solver_command(solver_command),
        Dict{String,String}(opt => "" for opt in solver_args),
        stdin,
        stdout,
        MOI.FileFormats.NL.SolFileResults(
            "Optimize not called.",
            MOI.OPTIMIZE_NOT_CALLED,
        ),
        NaN,
    )
end

Base.show(io::IO, ::Optimizer) = print(io, "An SDPA model")

function MOI.empty!(model::Optimizer)
    MOI.empty!(model.inner)
    model.results = MOI.FileFormats.NL.SolFileResults(
        "Optimize not called.",
        MOI.OPTIMIZE_NOT_CALLED,
    )
    model.solve_time = NaN
    return
end

MOI.is_empty(model::Optimizer) = MOI.is_empty(model.inner)

MOI.get(model::Optimizer, ::MOI.SolverName) = "SDPA"

MOI.supports(::Optimizer, ::MOI.Name) = true

MOI.get(model::Optimizer, ::MOI.Name) = MOI.get(model.inner, MOI.Name())

function MOI.set(model::Optimizer, ::MOI.Name, name::String)
    MOI.set(model.inner, MOI.Name(), name)
    return
end

MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

function MOI.get(model::Optimizer, attr::MOI.RawOptimizerAttribute)
    return get(model.options, attr.name, nothing)
end

function MOI.set(model::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    model.options[attr.name] = value
    return
end

const _SCALAR_FUNCTIONS = Union{
    MOI.VariableIndex,
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64},
}

const _SCALAR_SETS = Union{
    MOI.LessThan{Float64},
    MOI.GreaterThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:_SCALAR_FUNCTIONS},
    ::Type{<:_SCALAR_SETS},
)
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VariableIndex},
    ::Type{<:Union{MOI.ZeroOne,MOI.Integer}},
)
    return false
end

MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{<:_SCALAR_FUNCTIONS}) = true

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

MOI.supports(::Optimizer, ::MOI.NLPBlock) = false

function MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex},
)
    return false
end

MOI.supports_incremental_interface(::Optimizer) = false

MOI.copy_to(dest::Optimizer, src::MOI.ModelLike) = MOI.copy_to(dest.inner, src)

function MOI.optimize!(model::Optimizer)
    start_time = time()
    temp_dir = mktempdir()
    sdpa_input_file = joinpath(temp_dir, "input.dat-s")
    open(io -> write(io, model.inner), sdpa_input_file, "w")
    options = String[isempty(v) ? k : "$(k)=$(v)" for (k, v) in model.options]
    try
        sol_file = call_solver(
            model.solver_command,
            sdpa_input_file,
            options,
            model.stdin,
            model.stdout,
        )
        # replace this with what eric wrote 
        model.results = MOI.FileFormats.NL.SolFileResults(sol_file, model.inner)
    catch err
        model.results = MOI.FileFormats.NL.SolFileResults(
            "Error calling the solver. Failed with: $(err)",
            MOI.OTHER_ERROR,
        )
    end
    model.solve_time = time() - start_time
    return
end

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time

function MOI.get(
    model::Optimizer,
    attr::Union{
        MOI.ResultCount,
        MOI.RawStatusString,
        MOI.TerminationStatus,
        MOI.PrimalStatus,
        MOI.DualStatus,
        MOI.ObjectiveValue,
    },
)
    return MOI.get(model.results, attr)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    x::MOI.VariableIndex,
)
    return MOI.get(model.results, attr, x)
end

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.ConstraintPrimal,MOI.ConstraintDual},
    ci::MOI.ConstraintIndex,
)
    return MOI.get(model.results, attr, ci)
end

function MOI.get(model::Optimizer, attr::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    # TODO(odow): replace this with the proper dual objective.
    return MOI.get(model, MOI.ObjectiveValue())
end
