# This is copied from the `build.jl` file from SDPA_GMP_Builder
# https://github.com/ericphanson/SDPA_GMP_Builder/releases/download/v7.1.2
function install_bb_sdpa_high_precision(prefix, verbose)
    products = Product[
        ExecutableProduct(prefix, "sdpa_gmp", :sdpa_gmp),
        ExecutableProduct(prefix, "sdpa_qd", :sdpa_qd),
        ExecutableProduct(prefix, "sdpa_dd", :sdpa_dd),
    ]
    
    # Download binaries from hosted location
    bin_prefix = "https://github.com/ericphanson/SDPA_GMP_Builder/releases/download/SDPA-QD-DD-GMP-rev-1"

    # Listing of files generated by BinaryBuilder:
    download_info = Dict(
        MacOS(:x86_64) => ("$bin_prefix/SDPA_GMP-QD-DD_Builder.v7.1.2.x86_64-apple-darwin14.tar.gz", "057a48faefbc2617fc1474d11287cc50ed6069e7d5c2d8c1a99cb75aa0746ad8"),
        Linux(:x86_64, libc=:glibc) => ("$bin_prefix/SDPA_GMP-QD-DD_Builder.v7.1.2.x86_64-linux-gnu.tar.gz", "00692deae34b52a72a9ee949ff227c2acc11eb31646e8c4fec418a9639620c53"),
    )
    
    # Install unsatisfied or updated dependencies:
    unsatisfied = any(!satisfied(p; verbose=verbose) for p in products)
    dl_info = choose_download(download_info, platform_key_abi())
    if dl_info === nothing && unsatisfied
        # If we don't have a compatible .tar.gz to download, complain.
        # Alternatively, you could attempt to install from a separate provider,
        # build from source or something even more ambitious here.
        error("Your platform (\"$(Sys.MACHINE)\", parsed as \"$(triplet(platform_key_abi()))\") is not supported by this package!")
    end
    
    # If we have a download, and we are unsatisfied (or the version we're
    # trying to install is not itself installed) then load it up!
    if unsatisfied || !isinstalled(dl_info...; prefix=prefix)
        # Download and install binaries
        install(dl_info...; prefix=prefix, force=true, verbose=verbose)
    end

    return products
end