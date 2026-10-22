# Revelar

App de **aprimoramento automático de fotos e vídeos** para o computador. Interface em português, modo escuro, processamento em segundo plano e exportação de **arquivos nativos para o iPhone** (não é um app da App Store).

Revelar não é o TrilhaPy nem o Inglês Real. Não pede login, não usa nuvem e não coloca marca d’água.

## O que ele faz

- Abre **JPG, PNG, HEIC** e **MP4 / MOV** (um ou vários arquivos, com fila visível).
- Fotos (resolução nativa): CLAHE no L do LAB (clip baixo, sem posterizar), unsharp de **alta frequência**, pretos/brancos automáticos por percentil (sombra preservada) e curva S para tirar o cinza lavado. JPEG q≈94 / PNG com sRGB.
- Vídeos: equalização de cor, estabilização leve e otimização de bitrate, num **único passe** de exportação.
- **100% automático:** sem slider de intensidade, sem barras de exposição, sem toque em “Aplicar”. Ao abrir um arquivo (ou as amostras), o pipeline profissional já corre em `threading.Thread`.
- A janela mostra **antes/depois**, progresso/status e a **pasta de saída**. Dá para enfileirar outro arquivo, cancelar o job e mudar o preview enquanto processa.

## Formato iPhone (arquivo, não IPA)

| Tipo  | Arquivo                         | Detalhes                                      |
|-------|---------------------------------|-----------------------------------------------|
| Foto  | `revelar-foto-001.jpg` ou `.png` | JPEG qualidade ≈ 94 ou PNG, perfil de cor sRGB |
| Vídeo | `revelar-video-001.mp4`          | **H.264** + áudio **AAC**, `yuv420p`, `+faststart` |

O Python **não instala no iOS**. Você gera o arquivo no PC e copia para o iPhone.

## Como executar

Requisitos: **Python 3.10+**. Para vídeo, o **ffmpeg** no PATH (ou o binário que vem com `imageio-ffmpeg`).

```bash
git clone https://github.com/alavancagem940-web/revelar.git
cd revelar
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

No Ubuntu/Debian, se faltar a interface gráfica:

```bash
sudo apt install python3-tk ffmpeg libheif1
```

No macOS: `brew install ffmpeg libheif`.

Amostras curtas ficam em `amostras/` (uma foto subexposta e um MP4 de 2 s). Na primeira abertura o app carrega essa pasta e **já revela** sozinho.

Pasta de saída padrão: `~/Revelar`. Você escolhe outra no topo da janela; JPEG (q≈94, sRGB) e MP4 H.264+AAC são gravados lá automaticamente.

## Como passar os arquivos para o iPhone

Não há IPA nem publicação na App Store. Use o arquivo exportado:

1. **AirDrop** — no Mac, envie o JPEG/PNG/MP4 para o iPhone; ele cai no app Fotos ou em Arquivos.
2. **Cabo** — ligue o iPhone no computador. No Mac, use o Finder (arquivos); no Windows, o Explorador. Copie para o aparelho e abra no app **Arquivos** ou **Fotos**.
3. **iCloud Drive** — grave na pasta do iCloud no PC; no iPhone, abra o app **Arquivos → iCloud Drive**.
4. **Arquivos** — qualquer nuvem ou pen drive visível no app Arquivos serve; toque o vídeo/foto para abrir ou salvar em Fotos.

O Fotos do iOS abre JPEG/PNG com sRGB e MP4 H.264/AAC sem converter de novo.

## Processamento em segundo plano

Toda revelação de foto e toda renderização de vídeo rodam em `threading.Thread`. A barra de progresso e o status (“lendo”, “CLAHE”, “exportando H.264”, percentual) atualizam ao vivo. Cancelar interrompe o job atual sem fechar o app. Erros (arquivo corrompido, codec ausente, disco cheio, ffmpeg faltando) aparecem na barra de status.

## Estrutura

- `main.py` — entrada
- `ui.py` — CustomTkinter (escuro, preview, fila, progressbar)
- `photo_engine.py` — OpenCV / Pillow (HEIC via pillow-heif)
- `video_engine.py` — MoviePy / OpenCV / ffmpeg
- `export_ios.py` — H.264/AAC e JPEG/PNG para iOS
- `jobs.py` — fila + thread + progresso

## Licença

Uso livre para o dono do repositório. Sem marca “alavancagem” nos arquivos exportados.
