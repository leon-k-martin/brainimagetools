.. _overview:

---------
Overview
---------
Biomedical knowledge about the brain increases every day, with a rapidly growing
number of scientific publications. While this informational plethora is not wholly
capturable by human beings, recent developments in information science and computational
linguistics aim to make this knowledge programmatically accessible by literature
mining. However, integrating these linguistic methods into neuroimaging standards
remains insufficient yet, hindering researchers from unraveling its full potential.
The semantic meta-analysis platform "The Virtual Brain adapter of Semantics" (TVBase)
introduces a new approach facilitating an anatomically integrated,
quantitative knowledge analysis for neuroscience, transforming results from the
literature-mining platform SCAIView (https://academia.scaiview.com) into neuroimaging data standards.

TVBase statistically extracts brain-related knowledge from over 36 million scientific articles from PubMed
and maps it on a template brain in MNI-space. TVBase assesses the association
strength between biomedical concepts and their associations with brain anatomy by measures of information
entropy. TVBase implements a unique transformation matrix between anatomical
terms of natural language and brain coordinates in standard brain parcellations
to foster a multimodal data and knowledge framework.

.. image:: imgs/Schematic1.png
  :width: 700
  :alt: TVBase_drawing


Biomedical Concepts in TVBase
==============================

The literature mining framework of TVBase is based on controlled vocabularies and semantic ontologies with
hierarchically defined concepts. Meaning that synonyms are systematically brought together
under unique identifiers pointing to an unambiguously defined concept.
Further, hierarchical definitions annotate sub-classes related to broader defined concepts (super-classes).
For example, the hippocampus is defined as part of the temporal lobe.
