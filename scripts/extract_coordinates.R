#!/usr/bin/env Rscript
# =============================================================================
# Extract genomic coordinates from SummarizedExperiment rowRanges
# Output: coordinates.tsv with ID, chr, start, end, width, strand
#
# Run: Rscript scripts/extract_coordinates.R
# =============================================================================

library(SummarizedExperiment)

rdata_dir <- "/Users/yousrabenzouari/Documents/Work_FMI/Manuscript_Draft/RData"
out_dir   <- "/Users/yousrabenzouari/Documents/Claude/Projects/Deeplearning or ML follow up project on my Nature Comm Paper/project3e_gnn_enriched_features/data/input"

cat("Loading SE object...\n")
se <- readRDS(file.path(rdata_dir,
  "03_SE_E8.5_E10.5_E14.5_genotype_MergedBioRep_WithCategories_GaussianThresh_Test4_after.rds"))

# Extract rowRanges (GRanges)
gr <- rowRanges(se)
coords <- as.data.frame(gr)

# Build output: keep ID (rownames), chr, start, end, width, strand
out <- data.frame(
  ID     = rownames(coords),
  chr    = coords$seqnames,
  start  = coords$start,
  end    = coords$end,
  width  = coords$width,
  strand = coords$strand,
  stringsAsFactors = FALSE
)

# Save
out_file <- file.path(out_dir, "coordinates.tsv")
write.table(out, out_file, sep = "\t", quote = FALSE, row.names = FALSE)
cat("Saved", nrow(out), "coordinates to", out_file, "\n")

# Quick summary
cat("\nSummary:\n")
cat("  Regions:", nrow(out), "\n")
cat("  Chromosomes:", length(unique(out$chr)), "\n")
cat("  Width range:", min(out$width), "-", max(out$width), "\n")
cat("  Median width:", median(out$width), "\n")
