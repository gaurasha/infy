//! `infy serve`: the OpenAI-compatible API over the native model (if any)
//! and every runtime.

use std::sync::Arc;

use infy_kernel::{Error, Result};
use infy_server::{router, serve, Server};

use crate::adapters::{Dispatch, SystemClock, SystemIds};
use crate::bootstrap::{Bootstrap, ModelArgs, SamplingArgs};
use crate::commands::wiring::{runtimes, target, Target};

pub fn run(
    boot: &Bootstrap,
    model: ModelArgs,
    listen: String,
    sampling: SamplingArgs,
) -> Result<()> {
    let defaults = sampling.to_params()?;
    let native = match model.model.as_deref() {
        Some(m) => match target(boot, Some(m))? {
            Target::Native(e) => Some(e),
            Target::Runtime(kind, name) => {
                return Err(Error::invalid(format!(
                    "--model {kind}:{name} names a runtime model; the server routes to runtimes by request model name, so start it without --model (or with a GGUF) and ask for {kind}:{name} per request"
                )))
            }
        },
        None => None,
    };
    let rt = Arc::new(runtimes(boot));
    let served: Vec<String> = {
        use infy_server::Backend;
        Dispatch::new(native.clone(), rt.clone())
            .models()
            .into_iter()
            .map(|m| m.id)
            .collect()
    };
    if served.is_empty() {
        eprintln!("warning: nothing to serve yet -- no --model and no runtime is running; requests will get 404 until one is");
    } else {
        eprintln!("serving: {}", served.join(", "));
    }
    let backend = Arc::new(Dispatch::new(native, rt));
    let server = Arc::new(Server::new(
        backend,
        Arc::new(SystemIds::default()),
        Arc::new(SystemClock::default()),
        defaults,
    ));
    eprintln!("\n  infy  ->  http://{listen}/v1\n");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| Error::wrap("starting the async runtime", e))?;
    runtime.block_on(serve(&listen, router(server)))
}
