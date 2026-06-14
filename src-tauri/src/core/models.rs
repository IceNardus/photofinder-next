use std::path::PathBuf;
use tracing::info;

/// Resolve model file path across platforms.
///
/// Priority:
/// 1. Relative to current executable (works for bundled apps on Windows/macOS)
/// 2. User's data directory (for manually placed models)
/// 3. Relative working directory (for development)
pub fn find_model_path(model_name: &str) -> PathBuf {
    // Strategy 1: relative to current executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            // Windows: exe is in install dir, models in resources/models subdir
            let candidate = exe_dir.join("resources").join("models").join(model_name);
            if candidate.exists() {
                info!("Found {} at: {}", model_name, candidate.display());
                return candidate;
            }

            // macOS: exe is .app/Contents/MacOS/, models in .app/Contents/Resources/resources/models/
            if let Some(contents_dir) = exe_dir.parent() {
                let candidate = contents_dir.join("Resources").join("resources").join("models").join(model_name);
                if candidate.exists() {
                    info!("Found {} at: {}", model_name, candidate.display());
                    return candidate;
                }
            }
        }
    }

    // Strategy 2: user data directory
    if let Some(data_dir) = dirs::data_local_dir() {
        let candidate = data_dir.join("PhotoFinderNext").join("resources").join("models").join(model_name);
        if candidate.exists() {
            info!("Found {} at: {}", model_name, candidate.display());
            return candidate;
        }
    }

    // Strategy 3: relative CWD (development)
    let candidate = PathBuf::from("resources/models").join(model_name);
    if candidate.exists() {
        info!("Found {} at: {}", model_name, candidate.display());
        return candidate;
    }

    // Fallback: return the exe-relative path even if it doesn't exist
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let fallback = exe_dir.join("resources").join("models").join(model_name);
            tracing::warn!("Model {} not found, using fallback: {}", model_name, fallback.display());
            return fallback;
        }
    }

    let fallback = PathBuf::from("resources/models").join(model_name);
    tracing::warn!("Model {} not found, using fallback: {}", model_name, fallback.display());
    fallback
}

/// Resolve the models directory across platforms.
pub fn find_models_dir() -> Option<PathBuf> {
    // Strategy 1: relative to current executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let candidate = exe_dir.join("resources").join("models");
            if candidate.exists() {
                info!("Found models dir at: {}", candidate.display());
                return Some(candidate);
            }

            // macOS bundle path
            if let Some(contents_dir) = exe_dir.parent() {
                let candidate = contents_dir.join("Resources").join("resources").join("models");
                if candidate.exists() {
                    info!("Found models dir at: {}", candidate.display());
                    return Some(candidate);
                }
            }
        }
    }

    // Strategy 2: user data directory
    if let Some(data_dir) = dirs::data_local_dir() {
        let candidate = data_dir.join("PhotoFinderNext").join("resources").join("models");
        if candidate.exists() {
            info!("Found models dir at: {}", candidate.display());
            return Some(candidate);
        }
    }

    // Strategy 3: relative CWD
    let candidate = PathBuf::from("resources/models");
    if candidate.exists() && candidate.join("scrfd_500m_bnkps.onnx").exists() {
        info!("Found models dir at: {}", candidate.display());
        return Some(candidate);
    }

    None
}
