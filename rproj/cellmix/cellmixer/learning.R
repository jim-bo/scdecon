# install CellMix (probably good to run as root)
#library(BiocInstaller)
#biocLite('CellMix', siteRepos = 'http://web.cbio.uct.ac.za/~renaud/CRAN')
#biocLite('GEOquery')

# load the libraries.
library(CellMix)
library(GEOquery)

# load the ACR data (both exp and cbc)
data <- gedData("GSE20300");
acr <- ExpressionMix('GSE20300', verbose = 1);
res <- gedBlood(acr, verbose = TRUE);

# extract the coefficients.
coef(res);

# extract the rraw data.
exprs(acr);

# save the sample names.
sampleNames(phenoData(acr));
