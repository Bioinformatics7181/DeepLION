This folder contains the data used in DeepLION: 

    ├── Example_raw_file.tsv        <- A raw TCR-sequencing data file example.
    │
    ├── THCA                        <- Processed files containing healthy individual samples as negative samples, named `Health_[NO.].tsv`, and thyroid cancer patient samples as positive samples, named `Patient_[NO.].tsv`.
    │                    
    ├── Lung                        <- Processed files containing healthy individual samples as negative samples, named `Health_[NO.].tsv`, and lung cancer patient samples as positive samples, named `Patient_[NO.].tsv`.
    │
    ├── TrainingSequences           <- Labeled TCR sequence data used to generate the reference dataset.
    │
    ├── Reference_dataset.tsv       <- Reference dataset, used in filtration process, containing TCR sequences appearing in both healthy individuals and cancer patients.
    │
    └── AAidx_PCA.txt               <- File recording a 20 × 15 amino acid feature matrix, obtained from Beshnova's work (Beshnova et al., 2020).  
