# SpeechEmotionRecognitionSystem
project work: Carlotta Giacchetta &amp; Chiara Musso 

Dataset aviable in:
1) IEMOCAP: https://www.kaggle.com/datasets/samuelsamsudinng/iemocap-emotion-speech-database?resource=download
2) RAVDESS: https://zenodo.org/records/1188976
3) BanglaSER: https://data.mendeley.com/datasets/t9h6p943xy/5



# abstract

The union of artificial intelligence and affective computing has led to groundbreaking advancements in understanding and interpreting human emotions. Among various communication methods, speech stands out as the most convenient, facilitating seamless exchanges between individuals and machines. While deep learning technology has propelled Speech Emotion Recognition (SER), challenges persist in effectively extracting emotional features from speech signals. This research explores the integration of features from different modalities, assessing various machine learning algorithms for emotion classification based on their acoustic correlation in speech utterances. The study utilizes a merged dataset from IEMOCAP, RADVESS, and BanglaSER, encompassing a diverse range of emotional expressions. Preprocessing techniques, including windowing, Fast Fourier Transform, Voice Activity Detector, and Z normalization, are employed to enhance the quality of speech signals. Feature extraction plays a c role, considering speech as a continuous signal of variable length that encapsulates both information and emotion. Mel-frequency cepstral coefficients (MFCCs), Linear Predictive Cepstral Coefficients (LPCC), intensity, autocorrelation pitch, and features based on the Teager energy operator are explored as effective representations of emotional information in speech. A critical step involves feature selection, where a sequential feature selection approach refines the dataset to 25 pertinent features, mitigating the risk of overfitting. Classification tasks employ Random Forest and Support Vector Machine (SVM) classifiers, chosen for their resilience and effectiveness in handling complex datasets. Applications in human-computer interaction, customer support, mental health assessment, empathetic AI, and educational quality assessments underscore the significance of accurate speech emotion recognition in diverse domains.
