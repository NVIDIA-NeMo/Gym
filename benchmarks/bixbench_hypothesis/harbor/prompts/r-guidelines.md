Guidelines for using the R programming language:
1. Load packages using this format to minimize verbose output:
   ```r
   if (!requireNamespace("package_name", quietly = TRUE)) {{
     install.packages("package_name")
   }}
   suppressPackageStartupMessages(library(package_name))
   ```
2. You must use the tidyverse wherever possible: dplyr, tidyr, ggplot2, readr, stringr, forcats, purrr, tibble, and lubridate.

3. All plots must be made using ggplot2. Here is an example of how to make a plot:
   #### Create a density scatter plot of FSC-A vs SSC-A
   ```r
    plot_data <- as.data.frame(dmso_data[, c("FSC-A", "SSC-A")])
    scatter_plot <- ggplot2::ggplot(plot_data, ggplot2::aes(x = `FSC-A`, y = `SSC-A`)) +
      ggplot2::geom_hex(bins = 100) +
      ggplot2::scale_fill_viridis_c(trans = "log10") +
      ggplot2::labs(
        title = "FSC-A vs SSC-A Density Plot (DMSO Control)",
        x = "FSC-A",
        y = "SSC-A"
      ) +
      ggplot2::theme_minimal()
   ```

3. Use explicit namespace qualification for functions. For example, use dplyr::select() instead of select().

4. For data operations, suppress messages about column name repairs:
   ```r
   variable_name <- read_excel("<fpath>.csv", col_names = FALSE, .name_repair = "minimal")
   ```
