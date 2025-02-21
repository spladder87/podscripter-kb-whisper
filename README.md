# KB-Whisper Replicate Deployment

[🇸🇪 Svenska](#kb-whisper-replicate-deployment-svenska) | [🇬🇧 English](#kb-whisper-replicate-deployment-english)

# KB-Whisper Replicate Deployment (Svenska)

Detta repository innehåller Cog-konfigurationen för att distribuera [KBLabs KB-Whisper Large](https://huggingface.co/KBLab/kb-whisper-large) modell till Replicate. Detta är samma kraftfulla taligenkänningsmodell som driver [Podscripter.se](https://podscripter.se), Sveriges ledande tjänst för ljudtranskribering.

## 🎧 Prova det på Podscripter.se

[Podscripter.se](https://podscripter.se) erbjuder professionell transkribering för:
- Podcasts
- Intervjuer
- Föreläsningar
- Möten
- Alla typer av ljud- och videofiler

### Varför välja Podscripter?
- ⚡ **Snabbt & Exakt** - Drivs av KB-Whisper, den mest exakta svenska taligenkänningsmodellen
- 💰 **Överkomliga Priser** - Från endast 0,42 kr per minut
- 🎯 **Specialanpassad** - Optimerad för svenskt innehåll
- 🔒 **Inget Konto Krävs** - Snabb transkribering tillgänglig direkt
- 💼 **Företagsfunktioner** - Värdepaket och extrafunktioner för regelbundna användare

### Value Packages
- 1 timme för endast 25 kr (0,42 kr/min)
- 3 timmar för endast 50 kr (0,28 kr/min)
- 8 timmar för endast 100 kr (0,21 kr/min)

## Om KB-Whisper

KB-Whisper är en toppmodern svensk taligenkänningsmodell utvecklad av Kungliga biblioteket. Den överträffar betydligt OpenAIs Whisper-modeller på svenskt innehåll och minskar ordfelfrekvensen (WER) med i genomsnitt 47% jämfört med whisper-large-v3.

### Prestandajämförelse (WER)

| Modellstorlek | Dataset | KB-Whisper | OpenAI Whisper |
|------------|---------|------------|----------------|
| large-v3   | FLEURS  | 5,4        | 7,8            |
|            | CommonVoice | 4,1     | 9,5            |
|            | NST     | 5,2        | 11,3           |

## Modelldetaljer

- Parametrar: 1,61B
- Träningsdata: Över 50 000 timmar svenskt ljud
- Licens: Apache 2.0
- Utvecklad av: Kungliga biblioteket / KBLab

## Användning på Replicate

Testa KB-Whisper på Replicate: [spladder87/kblab-whisper-diarization](https://replicate.com/spladder87/kblab-whisper-diarization)

Modellen körs på Nvidia L40S GPU-hårdvara. Du kan använda den genom:
- Replicates webbgränssnitt
- Replicate API
- Kommandoraden

[Ytterligare distributionsinstruktioner kommer att läggas till]

## Lokal Utveckling

För att installera och köra modellen lokalt, följ dessa steg:

1. **Klona Repositoryt**
   ```bash
   git clone https://github.com/spladder87/podscripter-kb-whisper.git
   cd podscripter-kb-whisper
   ```

2. **Installera Cog**
   ```bash
   curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
   chmod +x /usr/local/bin/cog
   ```

3. **Ladda ner Modellen**
   Först, ladda ner modellen från Hugging Face:
   ```bash
   python download_model.py
   ```

4. **Konvertera till CTranslate2-format**
   Konvertera den nedladdade modellen till CTranslate2-format för optimerad inferens:
   ```bash
   pip install ctranslate2>=3.15
   ct2-transformers-converter --model KBLab/kb-whisper-large \
     --output_dir /src/kb-whisper-large-ct2 \
     --copy_files tokenizer.json --force
   ```

5. **Bygg Containern**
   ```bash
   cog build
   ```

6. **Kör Prediktioner**
   ```bash
   cog predict -i audio=@path/to/your/audio.mp3
   ```

Modellen kommer att bearbeta din ljudfil och returnera transkriptionsresultaten.

### Testa Olika Ljudformat

Du kan testa modellen med olika ljudformat:
- MP3-filer
- WAV-filer
- M4A-filer
- De flesta andra vanliga ljudformat

### Systemkrav

- GPU med CUDA-stöd
- Minst 16GB GPU-minne
- Tillräckligt diskutrymme för modellen (cirka 6GB)

## Gör ditt innehåll tillgängligt

Över 430 miljoner människor världen över har nedsatt hörsel. Genom att tillhandahålla transkriptioner gör du ditt innehåll tillgängligt för alla som vill ta del av det, oavsett hur de bäst konsumerar det.

Besök [Podscripter.se](https://podscripter.se) för att börja transkribera ditt ljud idag!

## Licens

Detta repository är licensierat under Apache 2.0-licensen, vilket matchar originalmodellens licens.

## Erkännanden

- KBLab för att ha skapat och öppet delat KB-Whisper-modellen
- Kungliga biblioteket för deras arbete med svenska AI-modeller

---

# KB-Whisper Replicate Deployment (English)

This repository contains the Cog configuration for deploying [KBLab's KB-Whisper Large](https://huggingface.co/KBLab/kb-whisper-large) model to Replicate. This is the same powerful speech recognition model that powers [Podscripter.se](https://podscripter.se), the leading Swedish audio transcription service.

## 🎧 Try it Live at Podscripter.se

[Podscripter.se](https://podscripter.se) offers professional-grade transcription for:
- Podcasts
- Interviews
- Lectures
- Meetings
- Any audio or video file

### Why Choose Podscripter?
- ⚡ **Fast & Accurate** - Powered by KB-Whisper, the most accurate Swedish speech recognition model
- 💰 **Affordable Pricing** - Starting at just $0.042 per minute
- 🎯 **Purpose-Built** - Optimized for Swedish content
- 🔒 **No Account Required** - Quick transcription service available instantly
- 💼 **Business Features** - Value packages and additional features for regular users

### Value Packages
- 1 hour for just $2.50 ($0.042/min)
- 3 hours for just $5.00 ($0.028/min)
- 8 hours for only $10.00 ($0.021/min)

## About KB-Whisper

KB-Whisper is a state-of-the-art Swedish speech recognition model developed by the National Library of Sweden. It significantly outperforms OpenAI's Whisper models on Swedish language content, reducing Word Error Rate (WER) by an average of 47% compared to whisper-large-v3.

### Performance Comparison (WER)

| Model Size | Dataset | KB-Whisper | OpenAI Whisper |
|------------|---------|------------|----------------|
| large-v3   | FLEURS  | 5.4        | 7.8            |
|            | CommonVoice | 4.1     | 9.5            |
|            | NST     | 5.2        | 11.3           |

## Model Details

- Parameters: 1.61B
- Training Data: Over 50,000 hours of Swedish audio
- License: Apache 2.0
- Developed by: National Library of Sweden / KBLab

## Usage on Replicate

Try out KB-Whisper on Replicate: [spladder87/kblab-whisper-diarization](https://replicate.com/spladder87/kblab-whisper-diarization)

The model runs on Nvidia L40S GPU hardware. You can use it through:
- The Replicate web interface
- Replicate API
- Command line

[Additional deployment instructions will be added]

## Local Development

To set up and run the model locally, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/spladder87/podscripter-kb-whisper.git
   cd podscripter-kb-whisper
   ```

2. **Install Cog**
   ```bash
   curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
   chmod +x /usr/local/bin/cog
   ```

3. **Download the Model**
   First, download the model from Hugging Face:
   ```bash
   python download_model.py
   ```

4. **Convert to CTranslate2 Format**
   Convert the downloaded model to CTranslate2 format for optimized inference:
   ```bash
   pip install ctranslate2>=3.15
   ct2-transformers-converter --model KBLab/kb-whisper-large \
     --output_dir /src/kb-whisper-large-ct2 \
     --copy_files tokenizer.json --force
   ```

5. **Build the Container**
   ```bash
   cog build
   ```

6. **Run Predictions**
   ```bash
   cog predict -i audio=@path/to/your/audio.mp3
   ```

The model will process your audio file and return the transcription results.

### Testing Different Audio Files

You can test the model with various audio formats:
- MP3 files
- WAV files
- M4A files
- Most other common audio formats

### Resource Requirements

- GPU with CUDA support
- At least 16GB of GPU memory
- Sufficient disk space for the model (approximately 6GB)

## Make Your Content Accessible

Over 430 million people worldwide have disabling hearing loss. By providing transcriptions, you make your content accessible to everyone who wants to enjoy it, regardless of how they best consume it.

Visit [Podscripter.se](https://podscripter.se) to start transcribing your audio today!

## License

This repository is licensed under the Apache 2.0 License, matching the original model's license.

## Acknowledgments

- KBLab for creating and open-sourcing the KB-Whisper model
- The National Library of Sweden for their work on Swedish language AI models 