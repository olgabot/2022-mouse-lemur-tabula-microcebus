# 2022-mouse-lemur-tabula-microcebus
Code for cross-species and uTAR sequence homology analyses for two Tabula Microcebus papers

## Paper 1 -- Figure 6a

The Tabula Microcebus Consortium et al. Tabula Microcebus: A transcriptomic cell atlas of mouse lemur, an emerging primate model organism. bioRxiv 2021.12.12.469460 (2021) doi:10.1101/2021.12.12.469460 ([biorxiv link](https://www.biorxiv.org/content/10.1101/2021.12.12.469460v2.full))

![image](https://user-images.githubusercontent.com/806256/189369914-c902c302-e062-4ffc-a73a-b3b2a42b233a.png)

The data behind this figure includes:

- Unifying cell types across human, mouse, and mouse lemur
- Subsetting scanpy objects to only shared cell types
- Subsetting scanpy objects on homologous genes
- Creating a single scanpy object of human, mouse, and mouse lemur
- Performing [Xi ($\Xi$)](https://github.com/czbiohub/xicor) correlation on bootstrapped trios of (human, mouse, mouse lemur) cells
- Plotting $\Delta\Xi$ correlations to compare human:lemur vs human:mouse


## Paper 2 -- Supplemental Figure 1f

The Tabula Microcebus Consortium et al. Mouse lemur transcriptomic atlas elucidates primate genes, physiology, disease, and evolution. bioRxiv 2022.08.06.503035 (2022) doi:10.1101/2022.08.06.503035 ([biorxiv link](https://www.biorxiv.org/content/10.1101/2022.08.06.503035v1.full))

![image](https://user-images.githubusercontent.com/806256/189370037-2d8e1f91-9f69-48e6-b69e-032c39f82b4c.png)

- nf-predictorthologs pipeline
  - Used [orpheum](https://github.com/czbiohub/orpheum) to translate reads from unannotated transcriptionally active regions (uTARs)
    - Orpheum bisects reads into either coding or non-coding
  - Annotated translated protein sequences using [DIAMOND](https://github.com/bbuchfink/diamond)
- Create Venn Diagram of overlap between region annotations
