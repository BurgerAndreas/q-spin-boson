from qutip import Qobj, Options, mesolve, sigmaz, expect, projection
import numpy as np
import matplotlib.pyplot as plt
import functools as ft  # XX = ft.reduce(np.kron, [A, B, C, D, E])
import os

import hamiltonian_matrix as hm
import convention as cn
import trotter_fcts as plotting
import global_variables as pms

lind_plot_folder = './evolution_plots/qutip/'

def expct(_dm, _op):
    _dm, _op = np.asarray(_dm), np.asarray(_op)
    return np.real(np.trace(np.matmul(_dm, _op)))


def expct_str(_dm, _op):
    return r'$\vert$' + str(round(np.trace(np.matmul(_dm, _op)))) + r'$\rangle$'


def spin_dm_str(_dm, _op):
    if abs(expct(_dm, _op) - 1) < 0.5:  # up = 1 = ex
        return r'$\vert \uparrow \rangle$' 
    else:
        return r'$\vert \downarrow \rangle$' 


def spin_occ_dm_str(_dm, sz_op1, ada_opp=None, sz_op2=None, sz_op3=None):
    _label_str = r'$\vert$'
    if abs(expct(_dm, sz_op1) - 1) < 0.5:  # up = 1 = ex
        _label_str += r'$\uparrow$' 
    else:
        _label_str += r'$\downarrow$'
    if ada_opp is not None:
        _label_str += str(round(expct(_dm, ada_opp)))
    for _addop in [sz_op2, sz_op3]:
        if _addop is not None:
            if abs(expct(_dm, _addop) - 1) < 0.5:  # up = 1 = ex
                _label_str += r'$\uparrow$' 
            else:
                _label_str += r'$\downarrow$'
    _label_str += r'$\rangle$'
    return _label_str


def def_name_qutip(model, paras, gamma, bos, dt, time_points, with_h=True, with_l=True):
    name_qutip = model + '_' + paras + '_' + str(round(gamma, 2)) + '_' + str(int(bos)) + '_' + str(dt)
    if abs(time_points[-1] - 2.) > .2: name_qutip += '_t' + str(time_points[-1])
    if not with_h: name_qutip += '_noh'
    if not with_l: name_qutip += '_nol'
    return name_qutip


# https://stackoverflow.com/questions/34669082/open-quantum-system-modelling
def lindblad_dissipation(gamma_em=1, sv=True, _axp=None):
    my_ham = hm.hamiltonian_as_matrix()
    # sb_h, _ = sb_hamiltonian_x()
    ham_0 = np.array([[0, 0], [0, 0]])
    ham = Qobj(ham_0)

    # time
    evo_t_list = plotting.evo_time_points

    l1 = cn.pm  # sigma minus
    l1 = Qobj(gamma_em * l1)

    # initial state
    i0 = np.kron(cn.gs0, cn.gs0)  # ground state
    i1 = np.kron(cn.ex1, cn.ex1)  # excited state
    rho0 = Qobj(i1)

    sz_inv = Qobj(np.eye(2) - cn.sz)
    if sv:
        e_ops = []
    else:
        e_ops = [sigmaz(), sz_inv]
    dm_qutip = mesolve(H=ham, rho0=rho0, tlist=evo_t_list, c_ops=[l1], e_ops=e_ops)

    if _axp == None:
        plt.clf()
        plt.cla()
        plt.close()
        fig, _ax = plt.subplots()
    else:
        _ax = _axp

    if not sv:
        for state in dm_qutip.expect:
            _ax.plot(evo_t_list, state)
            _ax.set_xlabel('Time', fontsize=pms.labelsize)
            _ax.set_ylabel('expectation value', fontsize=pms.labelsize)
        if pms.shwplts: plt.show()
    else:
        _ax.plot(evo_t_list, np.real(expect(dm_qutip.states, projection(2, 0, 0))), label=r'$\rho_{11}$')
        _ax.plot(evo_t_list, expect(dm_qutip.states, projection(2, 1, 1)), label=r'$\rho_{22}$')
        _ax.title('Diagonal elements of density matrix')
        _ax.set_xlabel('Time', fontsize=pms.labelsize)
        _ax.set_ylabel('probabilities', fontsize=pms.labelsize)
        if _axp == None:
            plt.legend(loc='right', bbox_to_anchor=(1.3, 0.5), ncol=1, fancybox=True, frameon=False)
            plt.tight_layout(pad=1)
            #if _ax.lines: plt.savefig(dpi=pms.pltdpi, fname=plot_folder + 'lindblad_dissipation_' + str(gamma_em) + pms.pltfile, format=pms.format)
            plt.show(dpi=400)

    return dm_qutip


# sb_1f, sb_2f, jc_2f
def lindblad_simulation(model='sb_1f', paras='dyn', gamma_em=1., sv=True, bos=4, with_h=True, full_dim=False, with_l=True,
                        time_points=None, show=True, initial=None, _axp=None):

    # time
    if time_points is None: time_points = np.arange(0, 2.1, 0.1)
    # plot folder
    lind_plot_folder = './evolution_plots/qutip/'
    if time_points[-1] > 2.1: lind_plot_folder += 'steady_state/' 
    lind_plot_folder += model + '/'
    for path in [lind_plot_folder]:
        if not os.path.exists(path): os.makedirs(path)

    boson_id = np.eye(bos)
    spin_id = np.eye(2)
    ada = np.zeros([bos, bos])
    for _b in range(bos):
        ada[_b, _b] = _b

    # hamiltonian, dissipation
    if model == 'twolevel':
        dim_h = 2
        ham = Qobj(0)
        sz_op = cn.sz
        with_h = False
        # Diffusion
        with_l = True
        l1 = cn.pm  # sigma minus
        l1 = Qobj(gamma_em * l1)
        llist = [l1]

    elif model in pms.sb_1f_allvars:
        dim_h = bos * 2
        sb_h = hm.hamiltonian_as_matrix(model=model, excitations=bos, paras=paras)
        # ham = Qobj(inpt=sb_h, dims=np.shape(sb_h)[0], shape=np.shape(sb_h))
        ham = Qobj(sb_h)
        sz_op = np.kron(cn.sz, boson_id)
        ada_op = np.kron(spin_id, ada)

        # Diffusion
        l1 = np.kron(cn.pm, boson_id)  # sigma minus
        l1 = Qobj(gamma_em * l1)
        llist = [l1]

    elif model in ['sb_2f', 'jcps_2f']: 
        dim_h = 2 * bos * 2
        sb_h = hm.hamiltonian_as_matrix(model=model, excitations=bos, paras=paras)
        ham = Qobj(sb_h)
        sz_op1 = np.kron(cn.sz, np.kron(boson_id, spin_id))
        sz_op2 = np.kron(spin_id, np.kron(boson_id, cn.sz))
        ada_op = np.kron(spin_id, np.kron(ada, spin_id))

        # Diffusion
        l1 = np.kron(cn.pm, np.kron(boson_id, spin_id))  # sigma minus
        l1 = Qobj(gamma_em * l1)
        l2 = np.kron(spin_id, np.kron(boson_id, cn.pm))  # sigma minus
        l2 = Qobj(gamma_em * l2)
        llist = [l1, l2]


    elif model in ['jcrhs_2f', 'jcrhs_3f', 'jcps_2f', 'jcps_3f']:
        if model == 'jcrhs_2f': dim_h = 4
        if model == 'jcrhs_3f': dim_h = 8
        if model == 'jcps_2f': dim_h = 16  # TODO jcps_2f bosons - 12 or 16?
        if model == 'jcps_3f': dim_h = 32
        jc_h = hm.hamiltonian_as_matrix(model=model, excitations=4, paras=paras)
        ham = Qobj(jc_h)
        llist = []

    if not with_l:
        llist = []

    """ initial state """
    s0_initial = cn.gs0  # ground state
    s1_initial = cn.ex1  # excited state
    b0_initial, b1_initial = np.zeros([bos]), np.zeros([bos])
    b0_initial[0] = 1
    b1_initial[1] = 1
    if initial is None:
        if model in pms.sb_1f_allvars:
            # 0, 1 x 0, 1, 0, 0
            sv_initial = np.kron(s1_initial, b0_initial)  # 1x8
            i0 = np.matmul(sv_initial.T, sv_initial)  # 8x8
        elif model in ['sb_2f']:
            sv_initial = np.kron(s1_initial, np.kron(b0_initial, s0_initial))  # 1x16
            i0 = np.matmul(sv_initial.T, sv_initial)  # 16x16
        elif model == 'jcrhs_2f':
            sv_initial = np.matrix([1, 0, 0, 0])  # 4
            i0 = np.matmul(sv_initial.T, sv_initial)  # 4x4
        elif model == 'jcrhs_3f':
            sv_initial = np.matrix([1, 0, 0, 0, 0, 0, 0, 0])  # 8
            i0 = np.matmul(sv_initial.T, sv_initial)  # 8x8
        elif model == 'jcps_2f':
            # F - H - F
            sv_initial = ft.reduce(np.kron, [s1_initial, b1_initial, s0_initial])
            i0 = np.matmul(sv_initial.T, sv_initial)  # 12x12
        elif model == 'jcps_3f':
            sv_initial = ft.reduce(np.kron, [s1_initial, s1_initial, s1_initial, b0_initial])
            i0 = np.matmul(sv_initial.T, sv_initial)  # 32x32
        elif model == 'twolevel':
            sv_initial = s1_initial
        i0 = np.matmul(sv_initial.T, sv_initial) 
        #print('QuTip initial:', sv_initial)
    else:
        i0 = np.matmul(np.asmatrix(initial).T, np.asmatrix(initial))
        #print('QuTip initial (passed):', initial)

    rho0 = Qobj(i0)

    """ label basis states """
    label_by_op = []
    for _basis in range(dim_h):
        _basis_vec = np.zeros([dim_h])
        _basis_vec[_basis] = 1
        _basis_vec = np.asmatrix(_basis_vec)
        _basis_dm = np.matmul(_basis_vec.T, _basis_vec)
        if model == 'twolevel':
            label_by_op.append(spin_occ_dm_str(_basis_dm, sz_op, None))
        elif model in pms.sb_1f_allvars:
            label_by_op.append(spin_occ_dm_str(_basis_dm, sz_op, ada_op))
        elif model == 'sb_2f':
            label_by_op.append(spin_occ_dm_str(_basis_dm, sz_op1, ada_op, sz_op2))
        elif model == 'jcps_2f':
            # F - H - F
            # sz_op1 = ft.reduce(np.kron, [cn.sz, boson_id, spin_id])
            # sz_op2 = ft.reduce(np.kron, [spin_id, boson_id, cn.sz])
            # ada_op = ft.reduce(np.kron, [spin_id, ada, spin_id])
            label_by_op.append(spin_occ_dm_str(_basis_dm, sz_op1, ada_op, sz_op2))
        elif model == 'jcps_3f':
            sz_op1 = ft.reduce(np.kron, [cn.sz, spin_id, spin_id, boson_id])
            sz_op2 = ft.reduce(np.kron, [spin_id, cn.sz, spin_id, boson_id])
            sz_op3 = ft.reduce(np.kron, [spin_id, spin_id, cn.sz, boson_id])
            ada_op = ft.reduce(np.kron, [spin_id, spin_id, spin_id, ada])
            label_by_op.append(spin_occ_dm_str(_basis_dm, sz_op1, ada_op, sz_op2, sz_op3))
        elif model == 'jcrhs_2f':
            label_by_op = [r'$\vert \downarrow \downarrow 2 \rangle$', r'$\vert \uparrow \downarrow 1 \rangle$',
                           r'$\vert \downarrow \uparrow 1 \rangle$', r'$\vert \uparrow \uparrow 0 \rangle$']
        elif model == 'jcrhs_3f':
            label_by_op = [r'$\vert \downarrow \downarrow \downarrow 3 \rangle$', r'$\vert \uparrow \downarrow \downarrow 2 \rangle$',
                           r'$\vert \downarrow \uparrow \downarrow 2 \rangle$', r'$\vert \downarrow \downarrow \uparrow 2 \rangle$',
                           r'$\vert \uparrow \uparrow \downarrow 1 \rangle$', r'$\vert \uparrow \downarrow \uparrow 1 \rangle$',
                           r'$\vert \downarrow \uparrow \uparrow 1 \rangle$', r'$\vert \uparrow \uparrow \uparrow 0 \rangle$']


    #print('label_by_op', label_by_op)

    """ name for plot"""
    if with_h and with_l:
        model_name = model + '_full'
    elif with_h:
        model_name = model + '_ham'
    elif with_l:
        model_name = model + '_diff'

    if not sv and model == 'needtotestagain':  
        # deprecated
        # operators (only for sb_1f)
        sz_sb = np.kron(cn.sz, boson_id)
        sz_sb = Qobj(sz_sb)
        sz_inv = np.eye(dim_h) - sz_sb
        sz_inv = Qobj(sz_inv)
        e_ops = [sigmaz(), sz_inv]
    else:
        e_ops = []

    if with_h:
        dm_qutip = mesolve(H=ham, rho0=rho0, tlist=time_points, c_ops=llist, e_ops=e_ops)
    else:
        # no Hamiltonian
        # something needs this
        if full_dim:
            dm_qutip = mesolve(H=Qobj(np.zeros([dim_h, dim_h])), rho0=rho0, tlist=time_points, c_ops=llist, e_ops=e_ops)
        # evolve_trotter needs this
        else:
            dim_h = 2
            rho0 = Qobj(np.matmul(s1_initial.T, s1_initial))
            dm_qutip = mesolve(H=Qobj(np.zeros([dim_h, dim_h])), rho0=rho0, tlist=time_points,
                             c_ops=[Qobj(gamma_em * cn.pm)], e_ops=e_ops)
    # dm_qutip = mesolve(H=ham, rho0=rho0, tlist=evo_t_list)


    """ Plot """
    # colors
    if model in ['jcps_2f']:
        csevo = pms.cs_evo_jc
    else:
        csevo = pms.cs_evo_default
    if _axp == None:
        plt.clf()
        plt.cla()
        plt.close()
        _fig, _ax = plt.subplots(nrows=1, ncols=1, figsize=(pms.figx, pms.figy)) 
    else:
        _fig = _axp.get_figure()
        _ax = _axp

    if not sv:
        for state in dm_qutip.expect:
            _ax.plot(time_points, state)
            _ax.set_xlabel(pms.xlabel_t, fontsize=pms.labelsize)
            _ax.set_ylabel(pms.ylabel_obs, fontsize=pms.labelsize)
        if pms.shwplts: plt.show()
    else:
        # plot diagonal elements
        lines_for_legend = []
        for st in range(dim_h):
            _l, = _ax.plot(time_points, np.real(expect(dm_qutip.states, projection(dim_h, st, st))),
                    c=csevo[st],
                    #label=r'$\rho_{' + str(st) + '}$')
                    #label=label_by_op[st]
                    label=None
                    )
            lines_for_legend.append(_l)
        # _ax.plot(evo_t_list, expect(dm_qutip.states, projection(2, 1, 1)), label=r'$\rho_{22}$')
        #_ax.title('Lindblad simulator')
        # settings 
        _ax.set_xlabel(pms.xlabel_t, fontsize=pms.labelsize)
        _ax.set_ylabel(pms.ylabel_state, fontsize=pms.labelsize)
        _ax.set_xticks(time_points, pms.get_xlabels(time_points), fontsize=pms.ticksize)

        fname = lind_plot_folder + 'qutip_' + model_name + '_' + str(bos) + '_' + paras + '_g' + str(gamma_em) 
        if len(time_points) > 1:
            fname += '_dt' + str(time_points[1])[-1] + '_t' + str(time_points[-1]) 
        else:
            fname += str(time_points[0])
        # Save extra fig with just legend
        pms.extra_legend_fig(fname, _lines=lines_for_legend, _labels=label_by_op)

        fname += pms.pltfile
        if _axp == None:
            #pms.set_plt('right', dim_h)
            pms.set_plt(_lgnd=False)
            pms.fix_subplots(_fig=_fig, _axes=_ax)
            if _ax.lines: _fig.savefig(dpi=pms.pltdpi, fname=fname, format=pms.pltformat)
            if show:
                if pms.shwplts: _fig.show()
            plt.clf()
            plt.cla()
            plt.close()

        # print('t = 0\n', dm_qutip.states[0].full())
        # print('t = 1\n', dm_qutip.states[1].full())
        # print('t = 2\n', dm_qutip.states[2].full())
        # print('final state\n', dm_qutip.states[-1].full())
        # print('0', dm_qutip.states[0].matrix_element(bra, ket))

        """ Save """
        dt = abs(time_points[0] - time_points[1])
        dm_qutip_evo = []
        for t_step, _ in enumerate(time_points):
            dm_qutip_evo.append(dm_qutip.states[t_step].data.toarray())
        dm_qutip_evo = np.asarray(dm_qutip_evo)
        name_qutip = def_name_qutip(model, paras, gamma_em, bos, dt, time_points, with_h, with_l)
        np.save(pms.log_dir + pms.evo_logs + name_qutip + '_' + 'dm_qutip_evo' + '.npy', dm_qutip_evo)

    return dm_qutip

