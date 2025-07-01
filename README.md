# ChelatePred
ChelatePred is a ML model that can learn from the structure and geometry of chelators to predict which metal they can bind to (Cobalt, Copper, Manganese, Zinc and others).

Class_split_withsilluwet.py inputs excel sheet metal_complexDB.xlsx with the name of GML files, and look for each GML file in the folder gml_files.
The GML files contain undirected quantum graph information of each metal chelator structure (Quantum mechanically calculated parameters, DOI: 10.1039/d2dd00129b). This quantum mechanics informed structures are used for describing the feature space. Class_split_withsilluwet.py also makes sure that there is minimal data leakage between test set and training set. 
