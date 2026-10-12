# VisionQC – Controle de qualidade visual de peças de fundição

Classificação de peças metálicas de fundição como **corretas** ou **defeituosas** a partir de imagens, usando transfer learning com MobileNetV2, fine-tuning e Grad-CAM para analisar onde o modelo olha.

Projeto desenvolvido como PEC 4 do curso de Inteligência Artificial (CEI).

## Problema

Em controle de qualidade industrial, uma peça defeituosa que passa como boa chega ao cliente. Por isso, a métrica que mais importa é o **recall da classe defeituosa**, e não apenas a accuracy.

## Dataset

- **Origem:** [Casting Product Image Data for Quality Inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) (Kaggle).
- **Classes originais:** `ok_front` (peças sem defeito) e `def_front` (peças defeituosas).
- **Imagens usadas:** 7.348 no total (3.137 corretas e 4.211 defeituosas), todas da versão `casting_data`, com resolução de 300×300 pixels em RGB.
- **Divisão:** 70% treino, 15% validação e 15% teste, feita separadamente em cada classe com semente fixa (42).

| Partição | Corretas | Defeituosas | Total |
|---|---|---|---|
| Treino | 2.195 | 2.947 | 5.142 |
| Validação | 470 | 631 | 1.101 |
| Teste | 472 | 633 | 1.105 |

O dataset não está neste repositório. Baixe-o pelo link acima, ou deixe o notebook baixá-lo com `kagglehub`.

## Método

1. **Pré-processamento:** reescala para [0, 1] e redimensionamento para 224×224. Augmentação só no treino: espelhamento horizontal, rotação de ±10° e zoom de 10%.
2. **Baseline com transfer learning:** MobileNetV2 pré-treinada no ImageNet, com a base congelada e uma cabeça de classificação (GlobalAveragePooling2D, Dense 64, Dropout 0,3 e saída sigmoid). Uma camada `Rescaling` converte a entrada para [-1, 1], como a MobileNetV2 espera.
3. **Fine-tuning:** as últimas 30 camadas da base são liberadas, com BatchNormalization congelada e learning rate de 1e-5.
4. **Grad-CAM:** mapa de calor sobre a última camada convolucional (7×7) para mostrar as regiões que mais pesaram na decisão.

Treinamento: cabeça por até 10 épocas e fine-tuning por até 10 épocas, ambos com EarlyStopping (paciência 3) sobre a `val_loss`.

## Resultados no conjunto de teste (1.105 imagens)

| Modelo | Accuracy | Precisão (defeituosa) | Recall (defeituosa) | F1 | Erros |
|---|---|---|---|---|---|
| Baseline (base congelada) | 0,995 | 1,000 | 0,992 | 0,996 | 5 |
| Fine-tuning (30 camadas) | 0,997 | 1,000 | 0,995 | 0,998 | 3 |

- No baseline, a matriz de confusão tem 472 verdadeiros negativos, 0 falsos positivos, 5 falsos negativos e 628 verdadeiros positivos.
- Todos os erros do baseline são falsos negativos: peças defeituosas classificadas como corretas.
- A diferença entre os dois modelos é de 2 erros em 1.105 imagens. Isso é ruído, e não permite afirmar que o fine-tuning é melhor. A conclusão honesta é que ele manteve o desempenho.
- No fine-tuning, o EarlyStopping parou na época 6 e restaurou os pesos da época 3 (menor `val_loss`, 0,0086).

### Grad-CAM

Nas imagens analisadas, o modelo concentra a atenção na região central da peça (miolo e anel interno). Nos erros, o mapa é mais espalhado e chega à borda e até ao fundo, o que sugere que o modelo nem sempre se apoia no defeito. A resolução 7×7 do mapa indica a região aproximada, não o contorno do defeito.

## Limitações

- **Divisão aleatória:** o dataset tem variações da mesma peça, então treino e teste podem conter imagens muito parecidas. Isso pode tornar os resultados otimistas.
- **Teste pequeno:** com 1.105 imagens e poucos erros, cada imagem pesa muito na métrica.
- **Uma semente:** a variação entre treinos diferentes não foi medida.
- **Sem localização:** o modelo classifica a peça, mas não aponta onde está o defeito.

## Como executar

1. Abra o notebook no Google Colab, com GPU ativada (Runtime → Change runtime type → GPU).
2. Execute as células em ordem. A célula de download usa `kagglehub`, então você precisa de uma conta no Kaggle se o download exigir autenticação.
3. A execução de referência levou cerca de 14 minutos no treino da cabeça e cerca de 8 minutos no fine-tuning.

Dependências: TensorFlow/Keras, scikit-learn, NumPy, Pandas, Matplotlib, Pillow e kagglehub. No Colab, a maioria já vem instalada.

## Estrutura

```
VisionQC/
├── README.md
└── VisionQC_controle_qualidade_visual_pecas_metalicas.ipynb
```

## Próximos passos

1. Refazer o teste com a pasta `test` oficial do dataset, para medir o efeito da repetição de peças.
2. Repetir o treino com várias sementes e comparar a média do baseline e do fine-tuning.
3. Anotar os defeitos com caixas e treinar um detector (YOLO), para localizar o defeito e comparar com as regiões do Grad-CAM.
4. Servir o modelo por uma API FastAPI em Docker.

## Autor

Enzo Badanai
