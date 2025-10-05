"""
ODE (Orbital Data Explorer) API Client for Lunar Data Collection
"""
import requests
import os
import json
import urllib.parse
from typing import Dict, List, Tuple, Optional
from pprint import pprint
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://oderest.rsl.wustl.edu/live2/"


class ODEApiClient:
    """Client for interacting with NASA's ODE REST API"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        
    def _extract_first_feature_from_json(self, data: Dict) -> Optional[Dict]:
        """
        Robust extractor for 'feature' JSON in various ODE response shapes.
        Returns the feature dict or None.
        """
        if isinstance(data, dict):
            # 1) ODEResults wrapper
            if "ODEResults" in data and isinstance(data["ODEResults"], dict):
                odr = data["ODEResults"]
                feats = odr.get("Features")
                if feats:
                    feat = None
                    if isinstance(feats, dict) and "Feature" in feats:
                        f = feats["Feature"]
                        feat = f[0] if isinstance(f, list) else f
                    elif isinstance(feats, list):
                        feat = feats[0]
                    elif isinstance(feats, dict):
                        feat = feats
                    if feat:
                        return feat
            # 2) top-level "Features"
            if "Features" in data:
                feats = data["Features"]
                if isinstance(feats, dict) and "Feature" in feats:
                    f = feats["Feature"]
                    return f[0] if isinstance(f, list) else f
                if isinstance(feats, list) and feats:
                    return feats[0]
            # 3) top-level "features"
            if "features" in data:
                feats = data["features"]
                if isinstance(feats, list) and feats:
                    return feats[0]
                if isinstance(feats, dict):
                    return feats
            # 4) direct "Feature"
            if "Feature" in data:
                f = data["Feature"]
                return f[0] if isinstance(f, list) else f
        return None

    def query_feature_bbox(
        self, 
        featurename: str, 
        odemetadb: str = "moon", 
        featureclass: str = "crater", 
        output: str = "JSON"
    ) -> Tuple[float, float, float, float, Dict, Dict]:
        """
        Query ODE for feature bounding box.
        
        Returns:
            (minlat, maxlat, west, east, raw_json, feature_dict)
        """
        params = {
            "query": "featuredata",
            "odemetadb": odemetadb,
            "featureclass": featureclass,
            "featurename": featurename,
            "output": output
        }
        
        logger.info(f"Querying feature bbox for: {featurename}")
        r = requests.get(self.base_url, params=params)
        r.raise_for_status()
        data = r.json()
        
        feat = self._extract_first_feature_from_json(data)
        if feat is None:
            raise RuntimeError(f"Feature not found: {featurename}")
        
        def _get_key(d, *keys):
            for k in keys:
                if k in d:
                    return d[k]
            return None
        
        minlat = _get_key(feat, "MinLat", "minlat", "MinLatitude", "minlatitude")
        maxlat = _get_key(feat, "MaxLat", "maxlat", "MaxLatitude", "maxlatitude")
        west = _get_key(feat, "WestLon", "westlon", "WestLongitude", "westlongitude")
        east = _get_key(feat, "EastLon", "eastlon", "EastLongitude", "eastlongitude")
        
        if minlat is None or maxlat is None or west is None or east is None:
            raise RuntimeError("Feature found but bbox keys missing")
        
        return float(minlat), float(maxlat), float(west), float(east), data, feat

    def query_image_products(
        self,
        minlat: float,
        maxlat: float,
        west: float,
        east: float,
        target: str = "moon",
        pt: str = "IMG",
        ihid: Optional[str] = None,
        iid: Optional[str] = None,
        output: str = "JSON",
        results: str = "opmft",
        limit: int = 100,
        loc: str = "f"
    ) -> Dict:
        """Query ODE for image products within bounding box"""
        params = {
            "query": "product",
            "target": target,
            "pt": pt,
            "minlat": str(minlat),
            "maxlat": str(maxlat),
            "westernlon": str(west),
            "easternlon": str(east),
            "loc": loc,
            "results": results,
            "output": output,
            "limit": str(limit)
        }

        if ihid:
            params["ihid"] = ihid
        if iid:
            params["iid"] = iid

        logger.info(f"Querying image products for bbox: ({minlat}, {maxlat}, {west}, {east})")
        r = requests.get(self.base_url, params=params)
        r.raise_for_status()
        return r.json()

    def list_product_files(self, products_json: Dict, show_first_n: int = 10) -> List[Dict]:
        """
        Extract Product_files from various JSON layouts.
        
        Returns:
            List of dicts with keys: pdsid, odeid, pt, file_type, file_name, url
        """
        files_out = []
        prod_entries = []
        
        if isinstance(products_json, dict):
            if "ODEResults" in products_json and isinstance(products_json["ODEResults"], dict):
                prods_obj = products_json["ODEResults"].get("Products") or products_json["ODEResults"].get("Product")
                if prods_obj:
                    if isinstance(prods_obj, dict) and "Product" in prods_obj:
                        prods = prods_obj["Product"]
                    else:
                        prods = prods_obj
                    if isinstance(prods, list):
                        prod_entries = prods
                    elif isinstance(prods, dict):
                        if "Product" in prods:
                            prod_entries = prods["Product"]
                            if isinstance(prod_entries, dict):
                                prod_entries = [prod_entries]
                        else:
                            prod_entries = [prods]
            if not prod_entries:
                if "Products" in products_json:
                    prods = products_json["Products"]
                    if isinstance(prods, dict) and "Product" in prods:
                        prod_entries = prods["Product"]
                    else:
                        prod_entries = prods
                elif "Product" in products_json:
                    prod_entries = products_json["Product"]
            if isinstance(prod_entries, dict):
                prod_entries = [prod_entries]
            if isinstance(products_json, list):
                prod_entries = products_json

        for p in prod_entries:
            if not isinstance(p, dict):
                continue
            pdsid = p.get("pdsid") or p.get("PDS_ID") or p.get("ProductId") or p.get("product_id")
            odeid = p.get("ode_id") or p.get("odeid") or p.get("ODE_ID")
            pt = p.get("pt") or p.get("Label_product_type") or p.get("product_type")
            pf = p.get("Product_files") or p.get("ProductFiles") or p.get("files") or {}
            pf_list = []
            if isinstance(pf, dict) and "Product_file" in pf:
                pf_list = pf["Product_file"]
            elif isinstance(pf, list):
                pf_list = pf
            elif isinstance(pf, dict):
                pf_list = [pf]
            
            for f in pf_list:
                if not isinstance(f, dict):
                    continue
                url = f.get("URL") or f.get("url") or f.get("FileURL") or f.get("File_Path")
                if isinstance(url, dict):
                    url = url.get("#text") or url.get("value") or None
                files_out.append({
                    "pdsid": pdsid,
                    "odeid": odeid,
                    "pt": pt,
                    "file_type": f.get("Type"),
                    "file_name": f.get("FileName") or f.get("File") or f.get("FileName"),
                    "url": url
                })
        
        logger.info(f"Found {len(files_out)} files (showing first {show_first_n})")
        for x in files_out[:show_first_n]:
            pprint(x)
        return files_out

    def download_url(
        self, 
        url: str, 
        outdir: str = "downloads", 
        fname: Optional[str] = None, 
        max_bytes: Optional[int] = None
    ) -> str:
        """Download a file from URL to local directory"""
        os.makedirs(outdir, exist_ok=True)
        if fname is None:
            fname = os.path.basename(urllib.parse.urlsplit(url).path) or "download"
        outpath = os.path.join(outdir, fname)
        
        logger.info(f"Downloading: {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(outpath, "wb") as fh:
                total = 0
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if not chunk:
                        break
                    fh.write(chunk)
                    total += len(chunk)
                    if max_bytes and total > max_bytes:
                        logger.info("Reached max_bytes, stopping early.")
                        break
        logger.info(f"Saved to: {outpath}")
        return outpath

    def collect_crater_data(
        self, 
        crater_name: str,
        output_dir: str = "raw_data",
        product_type: str = "CDRWAC4",
        max_products: int = 200
    ) -> Dict:
        """
        Complete workflow to collect data for a specific crater.
        
        Returns:
            Dictionary with metadata and file paths
        """
        # Query crater bbox
        minlat, maxlat, west, east, raw_json, feat = self.query_feature_bbox(
            crater_name, odemetadb="moon", featureclass="crater"
        )
        
        # Convert to -180/180 longitude system
        west_180 = west - 360.0 if west > 180 else west
        east_180 = east - 360.0 if east > 180 else east
        
        logger.info(f"Crater BBox - Lat: [{minlat}, {maxlat}], Lon: [{west_180}, {east_180}]")
        
        # Query image products
        products = self.query_image_products(
            minlat, maxlat, west_180, east_180,
            target="moon",
            pt=product_type,
            ihid="LRO",
            iid="LROC",
            results="opmft",
            output="JSON",
            limit=max_products,
            loc="i"
        )
        
        # Save metadata
        os.makedirs(output_dir, exist_ok=True)
        metadata_path = os.path.join(output_dir, f"{crater_name}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump({
                "crater": crater_name,
                "bbox": {
                    "minlat": minlat,
                    "maxlat": maxlat,
                    "west": west_180,
                    "east": east_180
                },
                "feature": feat,
                "products": products
            }, f, indent=2)
        
        logger.info(f"Saved metadata to: {metadata_path}")
        
        # Extract and download files
        files = self.list_product_files(products, show_first_n=30)
        
        # Download thumbnails
        thumbs = [f for f in files if f.get("file_type") and 
                 ("thumb" in str(f["file_type"]).lower() or "browse" in str(f["file_type"]).lower())]
        
        downloaded_files = []
        for t in thumbs[:5]:
            if t.get("url"):
                safe_name = f"thumb_{t.get('pdsid') or 'id'}_{os.path.basename(t['url'])}"
                fpath = self.download_url(t["url"], outdir=output_dir, fname=safe_name, max_bytes=2*1024*1024)
                downloaded_files.append(fpath)
        
        return {
            "crater": crater_name,
            "bbox": {"minlat": minlat, "maxlat": maxlat, "west": west_180, "east": east_180},
            "metadata_path": metadata_path,
            "downloaded_files": downloaded_files,
            "total_files": len(files)
        }