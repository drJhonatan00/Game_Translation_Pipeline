import os
import re
from typing import Dict, List, Tuple

from transformers import AutoTokenizer
import ctranslate2


GENGO_SENTAKU = {
    "1": ("Brazilian Portuguese", "Helsinki-NLP/opus-mt-en-pt", "modelo_opus_en_pt_ct2", None),
    "2": ("Spanish", "Helsinki-NLP/opus-mt-en-es", "modelo_opus_en_es_ct2", None),
    "3": ("French", "Helsinki-NLP/opus-mt-en-fr", "modelo_opus_en_fr_ct2", None),
    "4": ("Italian", "Helsinki-NLP/opus-mt-en-it", "modelo_opus_en_it_ct2", None),
    "5": ("German", "Helsinki-NLP/opus-mt-en-de", "modelo_opus_en_de_ct2", None),
    "6": ("Japanese", "Helsinki-NLP/opus-mt-en-jap", "modelo_opus_en_jap_ct2", None),
    "7": ("Simplified Chinese", "Helsinki-NLP/opus-mt-en-zh", "modelo_opus_en_zh_hans_ct2", "cmn_Hans"),
    "8": ("Traditional Chinese", "Helsinki-NLP/opus-mt-en-zh", "modelo_opus_en_zh_hant_ct2", "cmn_Hant"),
    "9": ("Korean", "Helsinki-NLP/opus-mt-tc-big-en-ko", "modelo_opus_en_ko_ct2", None),
    "10": ("Arabic", "Helsinki-NLP/opus-mt-en-ar", "modelo_opus_en_ar_ct2", None),
    "11": ("Polish", "Helsinki-NLP/opus-mt-en-pl", "modelo_opus_en_pl_ct2", None),
}

ROTO_TAMAN = 64
KIDOU_KIKI = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

HOGO_KEISHIKI = re.compile(
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


def gengo_sentaku_menu() -> Tuple[str, str, str, str]:
    print("=" * 70)
    print("           LOCALIZATION PIPELINE TOOL")
    print("                 LANGUAGE SELECTION")
    print("=" * 70)

    for kagi, deeta in GENGO_SENTAKU.items():
        print(f" [{kagi}] English -> {deeta[0]}")

    print("=" * 70)

    while True:
        sentaku = input("Choose the language (1-11): ").strip()

        if sentaku in GENGO_SENTAKU:
            namae, moderu_mei, ct2_foluda, chuugoku_code = GENGO_SENTAKU[sentaku]
            print(f"\nSelected language: {namae}")

            if chuugoku_code:
                print(f"Language code: {chuugoku_code}")

            return namae, moderu_mei, ct2_foluda, chuugoku_code

        print("\nInvalid! Choose a number between 1 and 11.\n")


def tsuuyaku_loader(
    moderu_mei: str,
    ct2_foluda: str
) -> Tuple[AutoTokenizer, ctranslate2.Translator]:

    print("\n" + "=" * 70)
    print("Loading the model")
    print("=" * 70)
    print(f"Model: {moderu_mei}")
    print(f"Device: {KIDOU_KIKI}")

    if not os.path.exists(ct2_foluda):
        from ctranslate2.converters import TransformersConverter

        print("\nCTranslate2 model not found.")
        print("Converting model to INT8...")
        print("This may take a while on the first run.\n")

        henkan_ki = TransformersConverter(moderu_mei)
        henkan_ki.convert(ct2_foluda, quantization="int8")

        print("Conversion Complete!")
    else:
        print("\nCTranslate2 model found.")
        print("Skipping conversion.")

    print("\nLoading tokenizer...")
    moji_bunseki = AutoTokenizer.from_pretrained(moderu_mei)

    print("Loading translator...")
    honyaku_ki = ctranslate2.Translator(ct2_foluda, device=KIDOU_KIKI)

    print("Model successfully loaded.")

    return moji_bunseki, honyaku_ki


def kakusu_gyou(gyou: str) -> Tuple[str, List[str]]:
    mitsuketa_tagu = []

    def kawaru(match_obj):
        moto_tagu = match_obj.group(0)
        bangou = len(mitsuketa_tagu)
        mitsuketa_tagu.append(moto_tagu)
        return f"TAGPLACEHOLDER{bangou}X"

    kakushita_gyou = HOGO_KEISHIKI.sub(kawaru, gyou)

    return kakushita_gyou, mitsuketa_tagu


def modoru_gyou(
    honyaku_zumi_gyou: str,
    moto_tagu: List[str]
) -> Tuple[str, bool]:

    kekka = honyaku_zumi_gyou
    zenbu_modotta = True

    for bangou, tagu in enumerate(moto_tagu):
        keishiki = re.compile(
            rf"TAG\s*PLACEHOLDER\s*{bangou}\s*X",
            re.IGNORECASE
        )

        if keishiki.search(kekka):
            kekka = keishiki.sub(
                lambda _: tagu,
                kekka,
                count=1
            )
        else:
            zenbu_modotta = False

    return kekka, zenbu_modotta


def tagu_kakunin(moto_tagu: List[str], saishuu_gyou: str) -> bool:

    for tagu in moto_tagu:
        if tagu not in saishuu_gyou:
            return False

    return True


def gyou_honyaku_dekiru(gyou: str) -> bool:
    if not gyou:
        return False

    if not any(moji.isalpha() for moji in gyou):
        return False

    return True


def txt_fairu_sagasu() -> List[str]:
    fairu_gun = []

    for fairu_mei in os.listdir("."):
        if not fairu_mei.lower().endswith(".txt"):
            continue

        if fairu_mei.startswith("translated_"):
            continue

        fairu_gun.append(fairu_mei)

    fairu_gun.sort()
    return fairu_gun


def shori_kumi_honyaku(
    bunshou_ichiran: List[str],
    moji_bunseki: AutoTokenizer,
    honyaku_ki: ctranslate2.Translator,
    chuugoku_code: str = None
) -> Dict[str, str]:

    if not bunshou_ichiran:
        return {}

    # CTranslate2 expects one token list per sentence: List[List[str]].
    fuda_batch = []

    for ichi_bunshou in bunshou_ichiran:
        if chuugoku_code:
            moderu_bunshou = f">>{chuugoku_code}<< {ichi_bunshou}"
        else:
            moderu_bunshou = ichi_bunshou

        ichi_fuda = moji_bunseki.convert_ids_to_tokens(
            moji_bunseki.encode(
                moderu_bunshou,
                add_special_tokens=True
            )
        )

        fuda_batch.append(ichi_fuda)

    kekka_ichiran = honyaku_ki.translate_batch(
        fuda_batch,
        max_batch_size=ROTO_TAMAN,
        batch_type="examples"
    )

    honyaku_map = {}

    for moto_bunshou, kekka in zip(bunshou_ichiran, kekka_ichiran):
        if not kekka.hypotheses:
            honyaku_map[moto_bunshou] = moto_bunshou
            continue

        honyaku_fuda = kekka.hypotheses[0]

        honyaku_id = moji_bunseki.convert_tokens_to_ids(
            honyaku_fuda
        )

        saki_bunshou = moji_bunseki.decode(
            honyaku_id,
            skip_special_tokens=True
        )

        honyaku_map[moto_bunshou] = saki_bunshou.strip()

    return honyaku_map


def lokaraizu_shori():
    txt_fairu_ichiran = txt_fairu_sagasu()

    if not txt_fairu_ichiran:
        print("\nNo .txt files found in this folder.")
        return

    print("\nFiles found:")
    for fairu_mei in txt_fairu_ichiran:
        print(f"  - {fairu_mei}")

    print(f"\nTotal: {len(txt_fairu_ichiran)} file(s)")

    gengo_mei, moderu_mei, ct2_foluda, chuugoku_code = gengo_sentaku_menu()

    moji_bunseki, honyaku_ki = tsuuyaku_loader(
        moderu_mei,
        ct2_foluda
    )

    print("\n" + "=" * 70)
    print("STEP 1/3 - ANALYZING FILES")
    print("=" * 70)

    honyaku_taishou = set()

    for fairu_mei in txt_fairu_ichiran:
        print(f"Analyzing: {fairu_mei}")

        try:
            with open(
                fairu_mei,
                "r",
                encoding="utf-8",
                errors="ignore",
                newline=""
            ) as nyuuryoku_fairu:
                for gyou in nyuuryoku_fairu:
                    naiyou = gyou.rstrip("\r\n")

                    if not gyou_honyaku_dekiru(naiyou):
                        continue

                    kakushita_gyou, _ = kakusu_gyou(naiyou)
                    honyaku_taishou.add(kakushita_gyou)

        except Exception as era:
            print(f"ERROR in the file {fairu_mei}: {era}")

    bunshou_ichiran = list(honyaku_taishou)

    print(f"\nTotal of {len(bunshou_ichiran)} unique lines found.")

    if not bunshou_ichiran:
        print("No translatable lines found.")
        return

    print("\n" + "=" * 70)
    print(f"STEP 2/3 - TRANSLATING FOR {gengo_mei.upper()}")
    print("=" * 70)

    honyaku_map = {}
    sou_suu = len(bunshou_ichiran)

    for kaishi in range(0, sou_suu, ROTO_TAMAN):
        owari = min(kaishi + ROTO_TAMAN, sou_suu)
        shori_kumi = bunshou_ichiran[kaishi:owari]

        shori_kumi_kekka = shori_kumi_honyaku(
            shori_kumi,
            moji_bunseki,
            honyaku_ki,
            chuugoku_code
        )

        honyaku_map.update(shori_kumi_kekka)

        shinchoku_ritsu = (owari / sou_suu) * 100
        print(
            f"Progress: {owari}/{sou_suu} "
            f"({shinchoku_ritsu:.1f}%)"
        )

    print("\n" + "=" * 70)
    print("STEP 3/3 - GENERATING FILES")
    print("=" * 70)

    sou_gyou = 0
    honyaku_sou_gyou = 0
    mondai_sou_gyou = 0

    for fairu_mei in txt_fairu_ichiran:
        atarashii_mei = f"translated_{fairu_mei}"
        print(f"\nProcessing: {fairu_mei}")

        try:
            with open(
                fairu_mei,
                "r",
                encoding="utf-8",
                errors="ignore",
                newline=""
            ) as nyuuryoku_fairu, open(
                atarashii_mei,
                "w",
                encoding="utf-8",
                newline=""
            ) as shutsuryoku_fairu:

                for gyou in nyuuryoku_fairu:
                    sou_gyou += 1

                    naiyou = gyou.rstrip("\r\n")
                    kaigyou = gyou[len(naiyou):]

                    if not gyou_honyaku_dekiru(naiyou):
                        shutsuryoku_fairu.write(gyou)
                        continue

                    kakushita_gyou, moto_tagu = kakusu_gyou(naiyou)

                    if kakushita_gyou not in honyaku_map:
                        shutsuryoku_fairu.write(gyou)
                        continue

                    honyaku_zumi_gyou = honyaku_map[kakushita_gyou]

                    saishuu_gyou, fukugen_zumi = modoru_gyou(
                        honyaku_zumi_gyou,
                        moto_tagu
                    )

                    tagu_seijou = tagu_kakunin(
                        moto_tagu,
                        saishuu_gyou
                    )

                    if not fukugen_zumi or not tagu_seijou:
                        shutsuryoku_fairu.write(gyou)
                        mondai_sou_gyou += 1
                        continue

                    shutsuryoku_fairu.write(
                        saishuu_gyou + kaigyou
                    )

                    honyaku_sou_gyou += 1

            print(f"OK -> {atarashii_mei}")

        except Exception as era:
            print(
                f"Process error {fairu_mei}: {era}"
            )

    print("\n" + "=" * 70)
    print("COMPLETE PROCESS")
    print("=" * 70)

    print(f"Language: {gengo_mei}")
    print(f"Device: {KIDOU_KIKI}")
    print(f"Processed Files: {len(txt_fairu_ichiran)}")
    print(f"Analyzed Lines: {sou_gyou}")
    print(f"Translated Lines: {honyaku_sou_gyou}")
    print(f"Security-protected lines: {mondai_sou_gyou}")

    if mondai_sou_gyou > 0:
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
            "All protected variables were "
            "restored correctly."
        )

    print("\nTranslated files were saved with the prefix:")
    print("translated_")
    print("=" * 70)


if __name__ == "__main__":
    try:
        lokaraizu_shori()

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by the user.")

    except Exception as era:
        print("\nFATAL ERROR:")
        print(era)

    finally:
        print("\nProgram concluded.\nThank you for using it!\nBy Dr. Jhonatan")
