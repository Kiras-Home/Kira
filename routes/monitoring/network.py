from flask import Blueprint, jsonify
import psutil
import time

# ✅ KORRIGIERT: Einfacher URL-Präfix
network_bp = Blueprint('network', __name__, url_prefix='/network')

@network_bp.route('/status', methods=['GET'])
def get_network_status():
    """Gibt den Netzwerkstatus zurück."""
    try:
        # Erste Messung
        net_io_1 = psutil.net_io_counters()
        time.sleep(1)  # 1 Sekunde warten
        # Zweite Messung für Geschwindigkeitsberechnung
        net_io_2 = psutil.net_io_counters()
        
        # Geschwindigkeiten berechnen (Bytes pro Sekunde)
        bytes_sent_per_sec = net_io_2.bytes_sent - net_io_1.bytes_sent
        bytes_recv_per_sec = net_io_2.bytes_recv - net_io_1.bytes_recv
        
        network_status = {
            'bytes_sent': net_io_2.bytes_sent,
            'bytes_recv': net_io_2.bytes_recv,
            'packets_sent': net_io_2.packets_sent,
            'packets_recv': net_io_2.packets_recv,
            'bytes_sent_per_sec': bytes_sent_per_sec,
            'bytes_recv_per_sec': bytes_recv_per_sec,
            'upload_speed_mbps': round((bytes_sent_per_sec * 8) / (1024**2), 2),
            'download_speed_mbps': round((bytes_recv_per_sec * 8) / (1024**2), 2),
            'total_sent_gb': round(net_io_2.bytes_sent / (1024**3), 2),
            'total_recv_gb': round(net_io_2.bytes_recv / (1024**3), 2)
        }
        
        return jsonify({'success': True, 'network_metrics': network_status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@network_bp.route('/interfaces', methods=['GET'])
def get_network_interfaces():
    """Gibt alle Netzwerk-Interfaces zurück"""
    try:
        interfaces = []
        net_if_addrs = psutil.net_if_addrs()
        net_if_stats = psutil.net_if_stats()
        
        for interface_name, addresses in net_if_addrs.items():
            interface_info = {
                'name': interface_name,
                'addresses': [],
                'is_up': net_if_stats[interface_name].isup if interface_name in net_if_stats else False,
                'speed': net_if_stats[interface_name].speed if interface_name in net_if_stats else 0
            }
            
            for addr in addresses:
                interface_info['addresses'].append({
                    'family': str(addr.family),
                    'address': addr.address,
                    'netmask': addr.netmask,
                    'broadcast': addr.broadcast
                })
                
            interfaces.append(interface_info)
            
        return jsonify({'success': True, 'interfaces': interfaces})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500