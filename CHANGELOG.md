# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.3.0a1] - 2025-11-27 (Alpha Release)

### Added
- **True just-in-time compilation**: compilation now occurs at loop execution time instead of function invocation, allowing the compiler to utilize full runtime context.
- **Automatic device detection and data movement**: runtime automatically selects the appropriate device and moves data as needed.
- **Automatic reduction detection**: identifies reduction operations for scalars and constant-indexing arrays.
- **Initial multi-backend code structure**: lays the groundwork for supporting multiple backends (Triton, CUDA, etc.).
- **GitHub CI/CD integration**: continuous integration and testing added for improved code quality.
- **Additional testing**: new tests added to cover core functionality and compiler workflows.

### Changed
- Refactored JIT and internal API to support runtime compilation and multi-backend extensibility.

### Notes
- This is an **alpha release**; some features are experimental and APIs may change in future versions.
- Feedback from early users is encouraged to help stabilize the final 0.3.0 release.
