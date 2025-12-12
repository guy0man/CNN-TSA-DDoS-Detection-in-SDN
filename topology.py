#!/usr/bin/env python3
"""
Mininet Topology for CNN-TSA DDoS Detection Testing
8 hosts, 1 switch, RYU controller with OpenFlow 1.3
"""

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink

def ddosTopology():
    """Create DDoS testing topology"""
    
    info('*** Creating network\n')
    net = Mininet(
        controller=RemoteController,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True
    )
    
    info('*** Adding controller\n')
    c0 = net.addController(
        'c0',
        controller=RemoteController,
        ip='127.0.0.1',
        port=6653,
        protocols='OpenFlow13'
    )
    
    info('*** Adding switch\n')
    s1 = net.addSwitch('s1', protocols='OpenFlow13')
    
    info('*** Adding hosts\n')
    # Victim host
    h1 = net.addHost('h1', ip='10.0.0.1/24', mac='00:00:00:00:00:01')
    
    # Benign hosts
    h2 = net.addHost('h2', ip='10.0.0.2/24', mac='00:00:00:00:00:02')
    h3 = net.addHost('h3', ip='10.0.0.3/24', mac='00:00:00:00:00:03')
    h4 = net.addHost('h4', ip='10.0.0.4/24', mac='00:00:00:00:00:04')
    h5 = net.addHost('h5', ip='10.0.0.5/24', mac='00:00:00:00:00:05')
    
    # Attacker hosts
    h6 = net.addHost('h6', ip='10.0.0.6/24', mac='00:00:00:00:00:06')
    h7 = net.addHost('h7', ip='10.0.0.7/24', mac='00:00:00:00:00:07')
    h8 = net.addHost('h8', ip='10.0.0.8/24', mac='00:00:00:00:00:08')
    
    info('*** Creating links\n')
    # Link all hosts to switch with bandwidth limits
    net.addLink(h1, s1, bw=100, delay='5ms')  # Victim
    net.addLink(h2, s1, bw=100, delay='5ms')  # Benign
    net.addLink(h3, s1, bw=100, delay='5ms')
    net.addLink(h4, s1, bw=100, delay='5ms')
    net.addLink(h5, s1, bw=100, delay='5ms')
    net.addLink(h6, s1, bw=100, delay='5ms')  # Attackers
    net.addLink(h7, s1, bw=100, delay='5ms')
    net.addLink(h8, s1, bw=100, delay='5ms')
    
    info('*** Starting network\n')
    net.build()
    c0.start()
    s1.start([c0])
    
    info('*** Testing connectivity\n')
    net.pingAll()
    
    info('*** Running CLI\n')
    CLI(net)
    
    info('*** Stopping network\n')
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    ddosTopology()