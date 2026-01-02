cp /content/drive/MyDrive/BigData/hw3/asset/Homo_sapiens_assembly38.fasta.gz* . 
cp /content/drive/MyDrive/BigData/hw3/asset/hashTable . 
cp /content/drive/MyDrive/BigData/hw3/asset/large_1.fastq.gz .
cp /content/drive/MyDrive/BigData/hw3/asset/large_2.fastq.gz .
mkdir -p kmers_index 
mv hashTable kmers_index/ 

N=50000
zcat large_1.fastq.gz | head -n $((N*4)) | gzip > small_1.fastq.gz
zcat large_2.fastq.gz | head -n $((N*4)) | gzip > small_2.fastq.gz