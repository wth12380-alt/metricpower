# =============================================================================
#  zzz.R  -  package load hook
#
#  The L3 layers add metrics to METRIC_REGISTRY. In script form that happened
#  at source() time; in a package the namespace is sealed after loading, so
#  registration must run inside .onLoad(), while the namespace is still
#  writable. Nothing else happens here.
# =============================================================================

.onLoad <- function(libname, pkgname) {
  ns <- asNamespace(pkgname)
  register_l3_cm(envir = ns)
  register_l3_cm_extra(envir = ns)
  invisible(NULL)
}

.onAttach <- function(libname, pkgname) {
  packageStartupMessage(sprintf(
    "metricpower: %d confusion-matrix metrics, %d L3 names across %d families, %d L4 pointers",
    length(get("METRIC_REGISTRY", envir = asNamespace(pkgname))),
    nrow(l3_manifest(print = FALSE)), length(L3_FAMILY), length(L4_REGISTRY)))
}
