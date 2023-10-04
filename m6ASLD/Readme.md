# Capturing short-range and long-range dependencies of nucleotides for identifying RNA N6-methyladenosine modification sites

**Abstract：** N6-methyladenosine (m6A) plays a crucial role in enriching RNA functional and genetic information, and the identification of m6A modification sites is therefore an important task to promote the understanding of RNA epigenetics. In the identification process, current studies are mainly concentrated on capturing the short-range dependencies between adjacent nucleotides in RNA sequences, while ignoring the impact of long-range dependencies between non-adjacent nucleotides for learning high-quality representation of RNA sequences. In this work, we propose an end-to-end prediction model, called M6ASLD, to improve the identification accuracy of m6A modification sites by capturing the short-range and long-range dependencies of nucleotides. Specifically, M6ASLD first encodes the type and position information of nucleotides to construct the initial embeddings of RNA sequences. A self-correlation map is then generated to characterize both short-range and long-range dependencies with a designed map generating block for each RNA sequence. After that, M6ASLD learns the global and local representations of RNA sequences by using a graph convolution process and a designed dependency searching block respectively, and finally achieves its identification task under a joint training scheme. Extensive experiments have demonstrated the promising performance of M6ASLD on three benchmark datasets across several evaluation metrics.





