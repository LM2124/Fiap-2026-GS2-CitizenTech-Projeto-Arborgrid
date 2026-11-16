# Guia Passo a Passo de Anotação (Labeling) — ArborGrid

> **Leia em 4 minutos antes de começar!**  
> Boa notícia: **você NÃO vai anotar do zero.** Todas as imagens já passaram por uma IA na GPU que desenhou a maior parte das árvores, postes e alguns cabos.  
> O seu único trabalho é **revisar rápido (~20 a 50 segundos por imagem)**: apagar sobras erradas e desenhar as "fitas" dos fios que faltaram.

---

## 1. Instalação e Configuração Inicial (2 Minutos, Sem Instalar Nada)

Usamos o **X-AnyLabeling** (gratuito, código aberto e portátil para Windows).

1. Acesse a página oficial de downloads:  
   👉 [https://github.com/CVHub520/X-AnyLabeling/releases](https://github.com/CVHub520/X-AnyLabeling/releases)
2. Na versão mais recente (*Latest*), baixe o executável **`X-AnyLabeling-CPU.exe`** (funciona direto em qualquer notebook Windows sem precisar configurar Python nem GPU).
3. Dê dois cliques para abrir o programa.
4. **PASSO OBRIGATÓRIO Nº 1 — Ligar Salvamento Automático:**
   * No menu superior, clique em **`File`** e certifique-se de que **`Auto Save`** está **marcado (`✔`)**. Assim você nunca perde nada ao trocar de foto!
5. **PASSO OBRIGATÓRIO Nº 2 — Abrir a Pasta de Trabalho:**
   * Aperte **`Ctrl + U`** (ou clique em **`File` -> `Open Dir`**).
   * Selecione a pasta que você recebeu (ex.: `pacote_colega_1_revised` ou `pacote_colega_2`).
   * **Pronto!** Os polígonos coloridos já vão aparecer automaticamente em cima das fotos. **NÃO clique em "Import" nem em "Export"!**

---

## 2. As Únicas 3 Etiquetas Permitidas

Existem **apenas 3 nomes** no projeto inteiro. Sempre escolha na listinha que aparece na tela (nunca invente nomes novos):

| Nome Exato (minúsculo, sem acento) | O que representa | Como deve ser desenhado |
| :--- | :--- | :--- |
| **`copa`** | Massa de folhas/galhos de **árvores** | Polígono envolvendo a copa |
| **`fiacao`** *(sem `ç` e sem `~`!)* | Cabos/fios aéreos nos postes | Polígono fino em formato de **"fita/faixa"** (~6 a 12 px de espessura) |
| **`poste`** | Postes de concreto da rede (e cruzeta) | Retângulo simples de 4 a 6 cliques cobrindo a coluna |

> **Nota sobre a caixinha "useDifficult":** Pode deixar sempre desmarcada! O formato YOLO de treinamento ignora essa opção completamente.

---

## 3. Regras Práticas — Como Rotular Rápido e Sem Erro

### ⚡ Regra 1: `fiacao` (Fios e Cabos — A MAIS IMPORTANTE!)
A pré-anotação acerta alguns fios contra o céu, mas **deixa passar vários fios** (especialmente na frente de muros e prédios) e às vezes confunde telhados com fios.

* ✅ **A "Regra dos 15 Pixels" (Agrupe feixes paralelos próximos):**
  * Quando houver vários fios paralelos colados (menos de ~15 pixels de distância entre si, como feixes de internet/telecomunicações): **NÃO desenhe fio por fio!** Desenhe **UMA ÚNICA fita** abraçando o feixe todo. Leva 5 segundos!
* ✅ **Quando separar em fitas individuais?**
  * Somente quando os cabos forem **claramente espaçados** com uma faixa larga de folhas verdes ou céu visível entre eles (ex.: os 3 cabos de Média Tensão / 13,8 kV no alto do poste, que ficam a 30–40 cm de distância uns dos outros). Desenhe uma fita estreita (~6 a 8 px) para cada fase.
* ✅ **Curvatura dos cabos (catelária):**
  * Não precisa de 15 cliques para contornar uma curva perfeita. **3 a 4 cliques ao longo da extensão** (início, uma curva suave no meio e o final) já geram uma aproximação poligonal excelente para a convolução da GPU.
* ✅ **Fios muito finos (drop cables / ramais solitários):**
  * Se o fio estiver no 1º ou 2º plano visível contra o céu ou galhos secos, **marque sim** com uma fita estreita de ~6 a 8 px (não se preocupe: nosso script de exportação auto-espessa polígonos finos para não sumirem no treino).
  * Só ignore fios "fantasmas" a 100 metros de distância lá no fundo da rua que viraram borrão indistinguível do céu.
* ✅ **REGRA CRÍTICA QUANDO O FIO ENTRA NA ÁRVORE (Copa Densa vs. Galhos Ralos):**
  * **Caso 1 — Copa densa e opaca (o fio some totalmente atrás de um bloco fechado de folhas):** desenhe uma fita até onde ele entra e **outra fita separada** onde ele reaparece do outro lado. Nunca chute um fio invisível por cima de uma massa fechada de folhas.
  * **Caso 2 — Galhos ralos / folhagem fina cortando o fio aos pedacinhos:** **NÃO pule o fio e NÃO faça 10 micro-polígonos picotados!** Passe **UMA ÚNICA fita contínua** acompanhando o cabo visível através desses galhinhos. Essa é justamente a zona de conflito físico mais valiosa para o modelo aprender!

```text
   CASO 1 — Copa densa fechada (2 fitas separadas onde o fio some por completo):
   =======[fita fiacao]=======> ( MASSA OPACA DE FOLHAS ) =======[fita fiacao]=======>

   CASO 2 — Galhos ralos/finos cortando o fio aos poucos (1 fita contínua!):
   =======[=========== fita única acompanhando o fio visível entre galhos ===========]=======>
```

* ❌ **DELETE falsos positivos em telhados:** Se a IA tiver marcado uma **telha de zinco**, beiral de telhado, calha, grade de portão ou antena como `fiacao`, clique nesse polígono e aperte **`Delete`**.
* ⚠️ **NÃO deixe fios óbvios sem marcar numa foto que você vai manter:** No formato YOLO, tudo que fica sem rótulo vira "fundo garantido (`0`)". Esquecer um fio evidente ensina a IA a ignorar fios reais!

---

### 🗼 Regra 2: `poste` (Postes de Concreto)
* ✅ **Qual poste marcar?** Postes de distribuição da concessionária (na calçada ou rente ao muro/portão, incluindo o poste particular padrão de entrada de concreto).
  * Tem formato de coluna de concreto cinza e segura a fiação da rua? Aperte `P` e faça um retângulo rápido de **4 cliques** cobrindo a coluna.
* ❌ **O que NÃO marcar como poste:**
  * Canos de ferro finos de 2 polegadas fixados no telhado/parede (pontaletes residenciais);
  * Mastros de antena de TV, para-raios, calhas ou grades de janela;
  * Postes minúsculos a 200 metros de distância lá no fundo da rua.

---

### 🌳 Regra 3: `copa` (Árvores)
* ✅ **Mantenha a margem folgada da IA:** A pré-anotação deixa uns pixels de folga em volta das folhas. **Pode deixar assim!** Não perca tempo contornando folha por folha.
* ✅ **Todas as árvores com potencial de tocar fios precisam de polígono.**
* 💡 **Arbustos baixos no chão/muro:** Se sobrarem alguns arbustos baixos perto da calçada marcados como `copa`, **não se preocupe em apagar** — como ficam longe da rede aérea no alto, não afetam o cálculo de risco de queda. Só apague trepadeiras/hera se estiverem subindo alto na parede encostando no ponto de entrada de energia.

---

### 🗑️ Regra 4: Foto Ruim, Cena "Embolada" ou Cansaço? DESCARTE!

* **O que é uma cena "embolada"?** Fotos com varal de roupa misturado com fiação, fachadas com grades densas de ferro, reflexo de sol estourado na lente, ou close excessivo da câmera a 1 metro das folhas (onde você até deduz onde o cabo passa, mas visualmente é impossível traçar sem cobrir 80% de folhas/galhos na fita).
* **Por que descartar é a melhor decisão técnica:** Em visão computacional, alimentar uma imagem embolada ensina a IA a confundir folhas e grades com fiação. Descartar **10 ou 15 fotos ruins** e manter **50 ou 60 fotos limpas e bem anotadas** gera um modelo de IA infinitamente mais preciso e robusto!
* **Como descartar em 1 segundo:** Aperte **`Ctrl + Shift + Delete`** e confirme: o programa move a foto e o JSON automaticamente para a pastinha `_delete_` e já pula para a próxima!

---

## 4. Cola Rápida de Atalhos no Teclado

| Tecla / Atalho | O que faz |
| :--- | :--- |
| **`D`** *(ou Seta Direita)* | **Próxima imagem** (já salva automaticamente!) |
| **`A`** *(ou Seta Esquerda)* | **Imagem anterior** |
| **`P`** *(ou `Ctrl + N`)* | **Criar novo polígono** (clique nos cantos; aperte `Enter` ou clique no 1º ponto para fechar) |
| **`Esc`** | **Cancela** o polígono que você estava desenhando |
| **`Ctrl + J`** | **Modo Editar / Selecionar** (para clicar num polígono errado, arrastar pontos ou apagar) |
| **`Delete`** | **Apaga** o polígono selecionado |
| **`Ctrl + Z`** | **Desfaz** o último clique ou exclusão acidental |
| **`Ctrl + Roda do Mouse`** | **Zoom In / Zoom Out** (ótimo para enxergar fios finos!) |
| **`Ctrl + Shift + Delete`** | **Descarta a imagem atual** (move para a lixeira segura `_delete_`) |

> **Dica de mestre para sair do modo desenhar:** Se você apertou `P` por engano, aperte **`Esc`** e em seguida **`Ctrl + J`** para voltar ao cursor normal de seleção!

---

## 5. Checklist de 10 Segundos Antes de Apertar `D` (Próxima Foto)

1. Tem telhado de zinco ou grade marcado errado como `fiacao`? → *`Ctrl + J` + clique + `Delete`.*
2. Faltou marcar algum feixe visível de fios passando na cena? → *`P` + 4 a 6 cliques em fita contínua + `fiacao`.*
3. A imagem está caótica, embolada demais ou vai te custar mais de 2 minutos? → *`Ctrl + Shift + Delete` e vá para a próxima sem culpa!*
4. Se estiver tudo certo: aperte **`D`** e continue o fluxo!

---

## 6. Como Consolidar Tudo Depois (Para o Treinamento YOLO-Seg)

Quando os pacotes revisados estiverem prontos, a consolidação final para treino do modelo na GPU roda com um único comando:

```powershell
ComputerVision\.venv\Scripts\python.exe ComputerVision\scripts\export_xanylabeling_to_yolo.py --merge-from pacote_colega_1_revised pacote_colega_2
```

O script:
* Consolida os JSONs revisados sem duplicar nada;
* Propaga automaticamente os descartes da pasta `_delete_`;
* Auto-espessa fios finos para ~5 px (garantindo que não sumam na máscara de saída do YOLO);
* Faz a divisão Treino (80%) / Validação (20%) agrupada por árvore para garantir **zero data leakage** entre ângulos;
* Gera o arquivo `data.yaml` pronto para rodar o treino na RTX 3060 Ti!
