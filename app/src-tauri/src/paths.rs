//! Where LocalFlow keeps its files.
//!
//! Everything the shell owns - settings, history, the log - lives in `%APPDATA%\LocalFlow`,
//! the same directory `localflow.config.CONFIG_DIR` uses, so the two halves of the app read
//! and write the same files.
//!
//! `APPDATA` is always set for a process started by a signed-in user, and reading it directly
//! was fine until something else started the shell: a launch through tooling handed it an
//! environment without the variable, and the shell then silently lost its settings, wrote its
//! history nowhere, and - worst - kept no log, which is the one thing that would have
//! explained the other two. It took a while to work out that the app was fine and the launch
//! was not. So derive the path from the profile when the variable is missing rather than
//! giving up on it.

use std::ffi::OsString;
use std::path::PathBuf;

fn non_empty(v: OsString) -> Option<OsString> {
    if v.is_empty() {
        None
    } else {
        Some(v)
    }
}

/// `%APPDATA%\LocalFlow`, or the same path derived from `%USERPROFILE%` when `APPDATA` is not
/// in the environment. `None` only if the process has neither, which means it has no user.
pub fn config_dir() -> Option<PathBuf> {
    if let Some(appdata) = std::env::var_os("APPDATA").and_then(non_empty) {
        return Some(PathBuf::from(appdata).join("LocalFlow"));
    }
    let profile = std::env::var_os("USERPROFILE").and_then(non_empty)?;
    Some(PathBuf::from(profile).join("AppData").join("Roaming").join("LocalFlow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both variables are set in any normal process, so this only checks the shape: the
    /// directory is named after the app and sits under a real profile.
    #[test]
    fn the_config_directory_is_under_the_users_profile() {
        let dir = config_dir().expect("a process with a user has a config directory");
        assert_eq!(dir.file_name().unwrap(), "LocalFlow");
        assert!(dir.parent().is_some());
    }
}
