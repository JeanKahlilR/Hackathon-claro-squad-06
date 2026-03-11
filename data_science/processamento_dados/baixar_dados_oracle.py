"""
Baixa dados do OCI Object Storage para a pasta local da VM.

Autenticacao: Instance Principal (nao usa ~/.oci/config).

Exemplos:
    python data_science/processamento_dados/baixar_dados_oracle.py ^
        --bucket dataflow-warehouse ^
        --prefix score_01/ --prefix score_02/

    python data_science/processamento_dados/baixar_dados_oracle.py ^
        --bucket dataflow-warehouse ^
        --object dados/base.parquet ^
        --dest C:\\tmp\\data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import oci
from loguru import logger


def configure_local_logger(verbose: bool = False) -> None:
    """Configura log no console para o script."""
    logger.remove()
    logger.add(
        sys.stdout,
        level="DEBUG" if verbose else "INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{function}</cyan> - <level>{message}</level>"
        ),
        colorize=True,
    )


def default_data_dir() -> Path:
    """Retorna <raiz-do-projeto>/data."""
    return Path(__file__).resolve().parents[2] / "data"


def normalize_prefix(prefix: str) -> str:
    """Normaliza prefix para busca em "pasta/"."""
    cleaned = prefix.strip().lstrip("/")
    if cleaned and not cleaned.endswith("/"):
        cleaned = f"{cleaned}/"
    return cleaned


def parse_args() -> argparse.Namespace:
    """Le argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Baixa objetos do OCI Object Storage para disco local."
    )
    parser.add_argument("--bucket", required=True, help="Nome do bucket no Object Storage.")
    parser.add_argument(
        "--namespace",
        help="Namespace do Object Storage. Se omitido, busca automaticamente.",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="Prefixo/pasta remota para baixar (pode repetir). Ex: score_01/",
    )
    parser.add_argument(
        "--object",
        action="append",
        default=[],
        help="Objeto especifico para baixar (pode repetir). Ex: dados/base.parquet",
    )
    parser.add_argument(
        "--dest",
        default=str(default_data_dir()),
        help="Diretorio de destino local. Padrao: <raiz-do-projeto>/data",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("OCI_REGION"),
        help="Regiao OCI (ex: sa-saopaulo-1). Padrao: usa OCI_REGION se definido.",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=8,
        help="Tamanho do chunk (MB) para stream de download. Padrao: 8",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescreve arquivos locais ja existentes.",
    )
    parser.add_argument(
        "--keep-prefix",
        action="store_true",
        help="Mantem a estrutura completa da chave remota ao salvar localmente.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="Limita quantidade de objetos (util para teste).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas lista o que seria baixado, sem gravar arquivos.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Ativa logs detalhados.",
    )

    args = parser.parse_args()
    if not args.prefix and not args.object:
        parser.error("Informe ao menos um --prefix ou --object.")
    if args.chunk_size_mb <= 0:
        parser.error("--chunk-size-mb precisa ser > 0.")
    return args


def build_object_storage_client(region: Optional[str]) -> oci.object_storage.ObjectStorageClient:
    """Cria cliente OCI autenticado por Instance Principal."""
    signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
    config = {"region": region} if region else {}
    client = oci.object_storage.ObjectStorageClient(config=config, signer=signer)

    if region:
        # Garante endpoint correto quando a regiao eh conhecida.
        client.base_client.set_region(region)

    return client


def resolve_namespace(
    client: oci.object_storage.ObjectStorageClient, explicit_namespace: Optional[str]
) -> str:
    """Resolve namespace manualmente ou via API."""
    if explicit_namespace:
        return explicit_namespace
    namespace = client.get_namespace().data
    if not namespace:
        raise RuntimeError("Nao foi possivel resolver namespace automaticamente.")
    return namespace


def list_objects_by_prefix(
    client: oci.object_storage.ObjectStorageClient,
    namespace: str,
    bucket: str,
    prefix: str,
    max_objects: Optional[int],
) -> List[Dict[str, Optional[object]]]:
    """Lista objetos do bucket para um prefixo com paginacao."""
    items: List[Dict[str, Optional[object]]] = []
    start: Optional[str] = None

    while True:
        response = client.list_objects(
            namespace_name=namespace,
            bucket_name=bucket,
            prefix=prefix,
            start=start,
            fields="name,size",
        )
        objects = response.data.objects or []

        for obj in objects:
            name = (obj.name or "").strip()
            if not name or name.endswith("/"):
                continue
            items.append(
                {
                    "name": name,
                    "size": obj.size,
                    "matched_prefix": prefix,
                }
            )
            if max_objects is not None and len(items) >= max_objects:
                return items

        start = response.data.next_start_with
        if not start:
            break

    return items


def collect_targets(
    client: oci.object_storage.ObjectStorageClient,
    namespace: str,
    bucket: str,
    prefixes: List[str],
    objects: List[str],
    max_objects: Optional[int],
) -> List[Dict[str, Optional[object]]]:
    """Consolida lista final de objetos para download, sem duplicados."""
    targets: List[Dict[str, Optional[object]]] = []
    seen: set[str] = set()

    for object_name in objects:
        cleaned = object_name.strip().lstrip("/")
        if not cleaned or cleaned in seen:
            continue
        targets.append({"name": cleaned, "size": None, "matched_prefix": None})
        seen.add(cleaned)

    for prefix in prefixes:
        normalized = normalize_prefix(prefix)
        if not normalized:
            continue

        listed = list_objects_by_prefix(
            client=client,
            namespace=namespace,
            bucket=bucket,
            prefix=normalized,
            max_objects=max_objects,
        )
        logger.info(f"Prefixo '{normalized}': {len(listed)} objeto(s) encontrado(s).")

        for item in listed:
            name = str(item["name"])
            if name in seen:
                continue
            targets.append(item)
            seen.add(name)
            if max_objects is not None and len(targets) >= max_objects:
                return targets

    return targets


def object_key_to_local_path(
    object_name: str,
    matched_prefix: Optional[str],
    destination_dir: Path,
    keep_prefix: bool,
) -> Path:
    """Converte chave remota para caminho local, evitando path traversal."""
    relative = object_name
    if not keep_prefix and matched_prefix and object_name.startswith(matched_prefix):
        stripped = object_name[len(matched_prefix) :].lstrip("/")
        if stripped:
            relative = stripped

    parts = [part for part in relative.split("/") if part and part != "."]
    if any(part == ".." for part in parts):
        raise ValueError(f"Objeto invalido com '..': {object_name}")
    if not parts:
        raise ValueError(f"Nao foi possivel resolver caminho local para: {object_name}")

    dest_root = destination_dir.resolve()
    local_path = (dest_root / Path(*parts)).resolve()

    if dest_root not in local_path.parents and local_path != dest_root:
        raise ValueError(f"Path traversal detectado para objeto: {object_name}")

    return local_path


def download_object_streaming(
    client: oci.object_storage.ObjectStorageClient,
    namespace: str,
    bucket: str,
    object_name: str,
    local_path: Path,
    chunk_size_bytes: int,
) -> int:
    """Baixa um objeto em stream para arquivo local."""
    response = client.get_object(
        namespace_name=namespace,
        bucket_name=bucket,
        object_name=object_name,
    )

    bytes_written = 0
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("wb") as file_handle:
        for chunk in response.data.raw.stream(chunk_size_bytes, decode_content=False):
            file_handle.write(chunk)
            bytes_written += len(chunk)

    return bytes_written


def main() -> int:
    """Ponto de entrada do script."""
    args = parse_args()
    configure_local_logger(verbose=args.verbose)

    destination_dir = Path(args.dest).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    chunk_size_bytes = args.chunk_size_mb * 1024 * 1024

    logger.info("Iniciando autenticacao via Instance Principal.")
    logger.info(f"Bucket: {args.bucket}")
    logger.info(f"Destino local: {destination_dir}")

    try:
        client = build_object_storage_client(region=args.region)
        namespace = resolve_namespace(client=client, explicit_namespace=args.namespace)
        logger.info(f"Namespace: {namespace}")
        if args.region:
            logger.info(f"Regiao: {args.region}")

        targets = collect_targets(
            client=client,
            namespace=namespace,
            bucket=args.bucket,
            prefixes=args.prefix,
            objects=args.object,
            max_objects=args.max_objects,
        )

        if not targets:
            logger.warning("Nenhum objeto encontrado para os filtros informados.")
            return 0

        logger.info(f"Total de objetos para processar: {len(targets)}")

        downloaded = 0
        skipped = 0
        failed = 0
        total_bytes = 0

        for index, item in enumerate(targets, start=1):
            object_name = str(item["name"])
            matched_prefix = item["matched_prefix"]

            try:
                local_path = object_key_to_local_path(
                    object_name=object_name,
                    matched_prefix=matched_prefix if isinstance(matched_prefix, str) else None,
                    destination_dir=destination_dir,
                    keep_prefix=args.keep_prefix,
                )
            except ValueError as exc:
                failed += 1
                logger.error(f"[{index}/{len(targets)}] {exc}")
                continue

            if local_path.exists() and not args.overwrite:
                skipped += 1
                logger.info(
                    f"[{index}/{len(targets)}] Pulando existente: "
                    f"{object_name} -> {local_path}"
                )
                continue

            logger.info(f"[{index}/{len(targets)}] Baixando: {object_name} -> {local_path}")
            if args.dry_run:
                continue

            try:
                bytes_written = download_object_streaming(
                    client=client,
                    namespace=namespace,
                    bucket=args.bucket,
                    object_name=object_name,
                    local_path=local_path,
                    chunk_size_bytes=chunk_size_bytes,
                )
                downloaded += 1
                total_bytes += bytes_written
                logger.success(
                    f"[{index}/{len(targets)}] OK ({bytes_written / (1024 * 1024):.2f} MB)"
                )
            except Exception as exc:  # pragma: no cover - depende de ambiente OCI
                failed += 1
                logger.error(f"[{index}/{len(targets)}] Falha ao baixar '{object_name}': {exc}")

        logger.info("Resumo final:")
        logger.info(f"  - Objetos totais: {len(targets)}")
        logger.info(f"  - Baixados: {downloaded}")
        logger.info(f"  - Pulados: {skipped}")
        logger.info(f"  - Falhas: {failed}")
        if not args.dry_run:
            logger.info(f"  - Total escrito: {total_bytes / (1024 * 1024):.2f} MB")

        return 1 if failed > 0 else 0

    except oci.exceptions.ServiceError as exc:
        logger.error(
            f"Erro de servico OCI (status={exc.status}, code={exc.code}): {exc.message}"
        )
        if not args.region:
            logger.error(
                "Se o erro estiver relacionado a endpoint/regiao, "
                "informe --region (ex: sa-saopaulo-1)."
            )
        return 2
    except Exception as exc:
        logger.exception(f"Erro inesperado: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
