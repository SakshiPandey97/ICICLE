#!/usr/bin/env python3

import os
import sys
import glob
import shutil
import subprocess
from pathlib import Path

def get_accessions_from_excel(excel_path: Path) -> list:
    """Read accessions from Excel file."""
    import pandas as pd
    df = pd.read_excel(excel_path, sheet_name="Accessions")
    return df["Accession"].tolist()


def download_with_datasets(accession: str, output_path: Path, temp_dir: Path) -> bool:
    """Download using NCBI datasets CLI."""
    try:
        zip_path = temp_dir / f"{accession}.zip"
        extract_dir = temp_dir / f"{accession}_temp"
        
        result = subprocess.run([
            "datasets", "download", "genome", "accession", accession,
            "--include", "genome",
            "--filename", str(zip_path)
        ], capture_output=True, timeout=180)
        
        if result.returncode != 0:
            return False
        
        subprocess.run([
            "unzip", "-o", "-q", str(zip_path), "-d", str(extract_dir)
        ], capture_output=True)
        
        fna_files = glob.glob(str(extract_dir / "ncbi_dataset" / "data" / accession / "*.fna"))
        if not fna_files:
            fna_files = glob.glob(str(extract_dir / "ncbi_dataset" / "data" / "*" / "*.fna"))
        
        if fna_files:
            shutil.move(fna_files[0], output_path)
        
        zip_path.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)
        
        return output_path.exists()
        
    except Exception as e:
        print(f"    datasets error: {e}")
        return False


def download_with_entrez(accession: str, output_path: Path) -> bool:
    """Download using Biopython Entrez."""
    try:
        from Bio import Entrez
        Entrez.email = \"your_email@example.com\"  # Set your NCBI Entrez email
        
        handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
        content = handle.read()
        handle.close()
        
        if content and len(content) > 100:
            with open(output_path, "w") as f:
                f.write(content)
            return True
        return False
        
    except Exception as e:
        print(f"    Entrez error: {e}")
        return False


def create_combined_fasta(genomes_dir: Path, output_path: Path):
    """Combine all downloaded genomes into a single FASTA file.
    
    Each genome gets a single header with its accession, and all contigs
    are concatenated (separated by 'N' spacers to avoid artificial k-mers).
    """
    fasta_files = sorted(genomes_dir.glob("*.fasta"))
    
    with open(output_path, "w") as outf:
        for fasta_path in fasta_files:
            accession = fasta_path.stem
            outf.write(f">{accession}\n")
            
            # Collect all sequences (may have multiple contigs)
            sequences = []
            current_seq = []
            
            with open(fasta_path) as inf:
                for line in inf:
                    if line.startswith(">"):
                        if current_seq:
                            sequences.append("".join(current_seq))
                            current_seq = []
                    else:
                        current_seq.append(line.strip())
                
                if current_seq:
                    sequences.append("".join(current_seq))
            
            # Join contigs with N spacer (avoids creating artificial k-mers at boundaries)
            full_seq = "NNNNNNNNNN".join(sequences)
            
            # Write in 80-char lines (standard FASTA format)
            for i in range(0, len(full_seq), 80):
                outf.write(full_seq[i:i+80] + "\n")
    
    print(f"Combined {len(fasta_files)} genomes into {output_path}")


def main():
    script_dir = Path(__file__).parent.resolve()
    excel_path = script_dir / "mobhunter_benchmark.xlsx"
    output_dir = script_dir / "MOBHunter_Data"
    genomes_dir = output_dir / "genomes"
    
    if not excel_path.exists():
        print(f"ERROR: Excel file not found: {excel_path}")
        print("Please place mobhunter_benchmark.xlsx in the same directory as this script.")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    genomes_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MOBHunter Benchmark Genome Download")
    print("=" * 60)
    
    accessions = get_accessions_from_excel(excel_path)
    print(f"Found {len(accessions)} accessions in Excel file")
    
    downloaded = {}
    failed = []
    
    for i, acc in enumerate(accessions, 1):
        fasta_path = genomes_dir / f"{acc}.fasta"
        
        if fasta_path.exists() and fasta_path.stat().st_size > 0:
            print(f"[{i:3d}/{len(accessions)}] {acc} - exists, skipping")
            downloaded[acc] = fasta_path
            continue
        
        print(f"[{i:3d}/{len(accessions)}] {acc} - downloading...", end=" ", flush=True)
        
        success = download_with_datasets(acc, fasta_path, genomes_dir)
        
        if not success:
            success = download_with_entrez(acc, fasta_path)
        
        if success and fasta_path.exists() and fasta_path.stat().st_size > 0:
            print("OK")
            downloaded[acc] = fasta_path
        else:
            print("FAILED")
            failed.append(acc)
    
    print("\n" + "=" * 60)
    print(f"Downloaded: {len(downloaded)}/{len(accessions)}")
    
    if failed:
        print(f"Failed ({len(failed)}): {failed}")
        failed_path = output_dir / "failed_accessions.txt"
        with open(failed_path, "w") as f:
            f.write("\n".join(failed))
        print(f"Failed accessions saved to: {failed_path}")
    
    # Create combined FASTA
    combined_fasta = output_dir / "mobhunter_genomes.fasta"
    print(f"\nCreating combined FASTA: {combined_fasta}")
    create_combined_fasta(genomes_dir, combined_fasta)
    
    # Print summary
    total_size = sum(f.stat().st_size for f in genomes_dir.glob("*.fasta"))
    combined_size = combined_fasta.stat().st_size
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Genomes downloaded: {len(downloaded)}")
    print(f"  Individual files: {genomes_dir}")
    print(f"  Combined FASTA: {combined_fasta}")
    print(f"  Total size: {total_size / 1e9:.2f} GB")
    print(f"  Combined size: {combined_size / 1e9:.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
