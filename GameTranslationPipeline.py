import os
import re
from typing import Dict, List, Tuple

from transformers import AutoTokenizer
import ctranslate2


OPCOES_IDIOMAS = {
    "1": ("Português (Brasil)", "Helsinki-NLP/opus-mt-en-pt", "modelo_opus_en_pt_ct2", None),
    "2": ("Espanhol", "Helsinki-NLP/opus-mt-en-es", "modelo_opus_en_es_ct2", None),
    "3": ("Francês", "Helsinki-NLP/opus-mt-en-fr", "modelo_opus_en_fr_ct2", None),
    "4": ("Italiano", "Helsinki-NLP/opus-mt-en-it", "modelo_opus_en_it_ct2", None),
    "5": ("Alemão", "Helsinki-NLP/opus-mt-en-de", "modelo_opus_en_de_ct2", None),
    "6": ("Japonês", "Helsinki-NLP/opus-mt-en-jap", "modelo_opus_en_jap_ct2", None),
    "7": ("Chinês Simplificado", "Helsinki-NLP/opus-mt-en-zh", "modelo_opus_en_zh_hans_ct2", "cmn_Hans"),
    "8": ("Chinês Tradicional", "Helsinki-NLP/opus-mt-en-zh", "modelo_opus_en_zh_hant_ct2", "cmn_Hant"),
    "9": ("Coreano", "Helsinki-NLP/opus-mt-tc-big-en-ko", "modelo_opus_en_ko_ct2", None),
}

TAMANHO_LOTE = 64
DEVICE = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

PADRAO_PROTEGIDO = re.compile(
    r"""
    (
        \[VAR[^\]]*\]
        |
        \[~\s*[0-9]+\]
        |
        \[[A-Za-z0-9_]+\]
        |
        \\[A-Za-z0-9]
        |
        /[A-Za-z0-9]
    )
    """,
    re.VERBOSE,
)


def menu_selecao_idioma() -> Tuple[str, str, str, str]:
    print("=" * 70)
    print("           LOCALIZATION PIPELINE TOOL")
    print("                 LANGUAGE SELECTION")
    print("=" * 70)

    for chave, dados in OPCOES_IDIOMAS.items():
        print(f" [{chave}] English -> {dados[0]}")

    print("=" * 70)

    while True:
        opcao = input("Choose the language (1-9): ").strip()

        if opcao in OPCOES_IDIOMAS:
            nome, model_name, ct2_dir, codigo_chines = OPCOES_IDIOMAS[opcao]
            print(f"\nSelected language: {nome}")

            if codigo_chines:
                print(f"Language code: {codigo_chines}")

            return nome, model_name, ct2_dir, codigo_chines

        print("\nInvalid! Choose a number between 1 and 9.\n")


def carregar_tradutor(
    model_name: str,
    ct2_dir: str
) -> Tuple[AutoTokenizer, ctranslate2.Translator]:

    print("\n" + "=" * 70)
    print("Loading the model")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Device: {DEVICE}")

    if not os.path.exists(ct2_dir):
        from ctranslate2.converters import TransformersConverter

        print("\nCTranslate2 model not found.")
        print("Converting model to INT8...")
        print("This may take a while on the first run.\n")

        converter = TransformersConverter(model_name)
        converter.convert(ct2_dir, quantization="int8")

        print("Conversion Complete!")
    else:
        print("\nCTranslate2 model found.")
        print("Skipping conversion.")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading translator...")
    translator = ctranslate2.Translator(ct2_dir, device=DEVICE)

    print("Model successfully loaded.")

    return tokenizer, translator


def mascarar_linha(linha: str) -> Tuple[str, List[str]]:
    tags_encontradas = []

    def substituir(match):
        tag_original = match.group(0)
        indice = len(tags_encontradas)
        tags_encontradas.append(tag_original)
        return f"TAGPLACEHOLDER{indice}X"

    linha_mascarada = PADRAO_PROTEGIDO.sub(substituir, linha)

    return linha_mascarada, tags_encontradas


def desmascarar_linha(
    linha_traduzida: str,
    tags_originais: List[str]
) -> Tuple[str, bool]:

    resultado = linha_traduzida
    todas_restauradas = True

    for indice, tag in enumerate(tags_originais):
        padrao = re.compile(
            rf"TAG\s*PLACEHOLDER\s*{indice}\s*X",
            re.IGNORECASE
        )

        if padrao.search(resultado):
            resultado = padrao.sub(
                lambda _: tag,
                resultado,
                count=1
            )
        else:
            todas_restauradas = False

    return resultado, todas_restauradas


def verificar_tags(
    tags_originais: List[str],
    linha_final: str
) -> bool:

    for tag in tags_originais:
        if tag not in linha_final:
            return False

    return True


def linha_pode_ser_traduzida(linha: str) -> bool:
    if not linha:
        return False

    if not any(caractere.isalpha() for caractere in linha):
        return False

    return True


def encontrar_arquivos_txt() -> List[str]:
    arquivos = []

    for arquivo in os.listdir("."):
        if not arquivo.lower().endswith(".txt"):
            continue

        if arquivo.startswith("translated_"):
            continue

        arquivos.append(arquivo)

    arquivos.sort()
    return arquivos


def traduzir_lote(
    textos: List[str],
    tokenizer: AutoTokenizer,
    translator: ctranslate2.Translator,
    codigo_chines: str = None
) -> Dict[str, str]:

    if not textos:
        return {}

    tokens_batch = []

    for texto in textos:
        if codigo_chines:
            texto_modelo = f">>{codigo_chines}<< {texto}"
        else:
            texto_modelo = texto

        tokens = tokenizer.convert_ids_to_tokens(
            tokenizer.encode(
                texto_modelo,
                add_special_tokens=True
            )
        )

        tokens_batch.append(tokens)

    resultados = translator.translate_batch(
        tokens_batch,
        max_batch_size=TAMANHO_LOTE
    )

    mapa_traducao = {}

    for texto_original, resultado in zip(textos, resultados):
        if not resultado.hypotheses:
            mapa_traducao[texto_original] = texto_original
            continue

        tokens_traduzidos = resultado.hypotheses[0]

        ids_traduzidos = tokenizer.convert_tokens_to_ids(
            tokens_traduzidos
        )

        texto_destino = tokenizer.decode(
            ids_traduzidos,
            skip_special_tokens=True
        )

        mapa_traducao[texto_original] = texto_destino.strip()

    return mapa_traducao


def processar_localizacao():
    arquivos_txt = encontrar_arquivos_txt()

    if not arquivos_txt:
        print("\nNo .txt files found in this folder.")
        return

    print("\nFiles found:")
    for arquivo in arquivos_txt:
        print(f"  - {arquivo}")

    print(f"\nTotal: {len(arquivos_txt)} file(s)")

    nome_idioma, model_name, ct2_dir, codigo_chines = menu_selecao_idioma()

    tokenizer, translator = carregar_tradutor(
        model_name,
        ct2_dir
    )

    print("\n" + "=" * 70)
    print("STEP 1/3 - ANALYZING FILES")
    print("=" * 70)

    linhas_para_traduzir = set()

    for arquivo in arquivos_txt:
        print(f"Analyzing: {arquivo}")

        try:
            with open(
                arquivo,
                "r",
                encoding="utf-8",
                errors="ignore",
                newline=""
            ) as f:
                for linha in f:
                    conteudo = linha.rstrip("\r\n")

                    if not linha_pode_ser_traduzida(conteudo):
                        continue

                    linha_mascarada, _ = mascarar_linha(conteudo)
                    linhas_para_traduzir.add(linha_mascarada)

        except Exception as erro:
            print(f"ERROR in the file {arquivo}: {erro}")

    lista_textos = list(linhas_para_traduzir)

    print(f"\nTotal of {len(lista_textos)} unique lines found.")

    if not lista_textos:
        print("No translatable lines found.")
        return

    print("\n" + "=" * 70)
    print(f"STEP 2/3 - TRANSLATING FOR {nome_idioma.upper()}")
    print("=" * 70)

    mapa_traducao = {}
    total = len(lista_textos)

    for inicio in range(0, total, TAMANHO_LOTE):
        fim = min(inicio + TAMANHO_LOTE, total)
        lote = lista_textos[inicio:fim]

        resultado_lote = traduzir_lote(
            lote,
            tokenizer,
            translator,
            codigo_chines
        )

        mapa_traducao.update(resultado_lote)

        porcentagem = (fim / total) * 100
        print(
            f"Progress: {fim}/{total} "
            f"({porcentagem:.1f}%)"
        )

    print("\n" + "=" * 70)
    print("STEP 3/3 - GENERATING FILES")
    print("=" * 70)

    total_linhas = 0
    total_traduzidas = 0
    total_problemas = 0

    for arquivo in arquivos_txt:
        novo_nome = f"translated_{arquivo}"
        print(f"\nProcessing: {arquivo}")

        try:
            with open(
                arquivo,
                "r",
                encoding="utf-8",
                errors="ignore",
                newline=""
            ) as f_in, open(
                novo_nome,
                "w",
                encoding="utf-8",
                newline=""
            ) as f_out:

                for linha in f_in:
                    total_linhas += 1

                    conteudo = linha.rstrip("\r\n")
                    quebra = linha[len(conteudo):]

                    if not linha_pode_ser_traduzida(conteudo):
                        f_out.write(linha)
                        continue

                    linha_mascarada, tags_originais = mascarar_linha(
                        conteudo
                    )

                    if linha_mascarada not in mapa_traducao:
                        f_out.write(linha)
                        continue

                    linha_traduzida = mapa_traducao[linha_mascarada]

                    linha_final, restaurado = desmascarar_linha(
                        linha_traduzida,
                        tags_originais
                    )

                    tags_ok = verificar_tags(
                        tags_originais,
                        linha_final
                    )

                    if not restaurado or not tags_ok:
                        f_out.write(linha)
                        total_problemas += 1
                        continue

                    f_out.write(
                        linha_final + quebra
                    )

                    total_traduzidas += 1

            print(f"OK -> {novo_nome}")

        except Exception as erro:
            print(
                f"Process error {arquivo}: {erro}"
            )

    print("\n" + "=" * 70)
    print("COMPLETE PROCESS")
    print("=" * 70)

    print(f"Language: {nome_idioma}")
    print(f"Device: {DEVICE}")
    print(f"Processed Files: {len(arquivos_txt)}")
    print(f"Analyzed Lines: {total_linhas}")
    print(f"Translated Lines: {total_traduzidas}")
    print(f"Security-protected lines: {total_problemas}")

    if total_problemas > 0:
        print("\nWARNING!")
        print(
            "Some lines encountered issues restoring "
            "the variables."
        )
        print(
            "These lines have been kept in the original language "
            "to avoid breaking the file."
        )
    else:
        print(
            "\nAll protected variables were "
            "restored correctly."
        )

    print("\nTranslated files were saved with the prefix:")
    print("translated_")
    print("=" * 70)


if __name__ == "__main__":
    try:
        processar_localizacao()

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by the user.")

    except Exception as erro:
        print("\nFATAL ERROR:")
        print(erro)

    finally:
        print("\nProgram concluded.\nThank you for use!\nBy Dr. Jhonatan")
