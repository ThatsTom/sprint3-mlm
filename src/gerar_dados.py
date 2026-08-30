from pathlib import Path
import numpy as np
import pandas as pd


def gerar_dataset(n=1400, seed=42):
    rng = np.random.default_rng(seed)

    temperatura = np.clip(rng.normal(28, 7, n), 5, 60)
    umidade = np.clip(rng.normal(52, 18, n), 5, 100)
    perda_pacotes = np.clip(rng.beta(1.5, 8, n) * 35, 0, 35)
    latencia = np.clip(rng.gamma(2.2, 35, n), 5, 450)
    tentativas_login_falhas = np.clip(rng.poisson(1.4, n), 0, 12)
    idade_firmware = np.clip(rng.gamma(2.0, 120, n), 1, 1200)
    consumo_cpu = np.clip(rng.beta(2.4, 2.2, n) * 100, 5, 100)
    trafego_saida = np.clip(rng.lognormal(5.7, 0.7, n), 20, 5000)

    criptografia = rng.choice(["sim", "nao"], n, p=[0.80, 0.20])
    protocolo = rng.choice(["MQTT", "CoAP", "HTTP"], n, p=[0.50, 0.25, 0.25])
    rede = rng.choice(["corporativa", "iot_isolada", "visitantes"], n, p=[0.40, 0.45, 0.15])
    tipo = rng.choice(["sensor", "camera", "gateway", "atuador"], n, p=[0.45, 0.20, 0.15, 0.20])

    logit = -3.4
    logit += 0.10 * np.maximum(temperatura - 30, 0)
    logit += 0.065 * np.maximum(35 - umidade, 0)
    logit += 0.14 * perda_pacotes
    logit += 0.012 * np.maximum(latencia - 80, 0)
    logit += 0.42 * tentativas_login_falhas
    logit += 0.0020 * idade_firmware
    logit += 0.022 * np.maximum(consumo_cpu - 70, 0)
    logit += 0.00045 * np.maximum(trafego_saida - 500, 0)
    logit += np.where(criptografia == "nao", 1.10, 0)
    logit += np.where(protocolo == "HTTP", 0.45, 0)
    logit += np.where(rede == "visitantes", 0.65, 0)
    logit += np.where(tipo == "camera", 0.20, 0)
    logit += rng.normal(0, 0.55, n)

    probabilidade = 1 / (1 + np.exp(-logit))
    risco_alto = rng.binomial(1, probabilidade)

    return pd.DataFrame({
        "temperatura_c": temperatura.round(2),
        "umidade_pct": umidade.round(2),
        "perda_pacotes_pct": perda_pacotes.round(2),
        "latencia_ms": latencia.round(2),
        "tentativas_login_falhas": tentativas_login_falhas.astype(int),
        "idade_firmware_dias": idade_firmware.round(0).astype(int),
        "consumo_cpu_pct": consumo_cpu.round(2),
        "trafego_saida_kbps": trafego_saida.round(2),
        "criptografia_ativa": criptografia,
        "protocolo": protocolo,
        "rede": rede,
        "tipo_equipamento": tipo,
        "risco_alto": risco_alto.astype(int),
    })


def main():
    raiz = Path(__file__).resolve().parents[1]
    destino = raiz / "data" / "dados_iot_ml.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    df = gerar_dataset()
    df.to_csv(destino, index=False)
    print(f"Dataset gerado em: {destino}")
    print(f"Linhas: {len(df)}")
    print(f"Proporcao de alto risco: {df['risco_alto'].mean():.2%}")


if __name__ == "__main__":
    main()
