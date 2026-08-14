fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "windows" {
        cc::Build::new()
            .file("src/win7_shim.c")
            .compile("win7_shim");

        println!("cargo:rustc-link-arg=/ALTERNATENAME:__imp_GetSystemTimePreciseAsFileTime=my_imp_GetSystemTimePreciseAsFileTime");

        let mut res = winres::WindowsResource::new();
        res.set_icon("src/icon.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=Failed to compile windows resource: {}", e);
        }
    }
}


