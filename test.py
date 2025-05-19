import re
import csv

sql = '''create table t_contract (
    id bigint auto_increment comment '主键',
    project_sub_id bigint not null comment '子立项ID',
    contract_num varchar(120) not null comment '合同编号',
    name varchar(200) not null comment '合同名称',
    contract_type varchar(5) not null comment '合同类型',
    opposite_side_type varchar(5) not null comment '向对方类型',
    opposite_side_id bigint not null comment '向对方id',
    status char default '2' not null comment '状态 0-生效中 1-已完成 2-待生效',
    contract_file varchar(2000) null comment '合同文件',
    attachment_url varchar(2000) null comment '附件地址',
    signing_time date not null comment '签约时间',
    signing_address varchar(300) not null comment '签约地点',
    signing_warehouse varchar(300) null comment '签约仓库，仓储类型由此字段',
    signing_money decimal(12,2) null comment '协议金额，仓储类型有此字段',
    transport_cost decimal(12,2) null comment '协议金额，运输类型有此字段',
    goods_id bigint null comment '商品id，销售、采购类型有此字段',
    pur_sale_num decimal(18,6) null comment '购销数量，销售、采购类型有此字段',
    unit_price decimal(12,2) null comment '单价，销售、采购类型有此字段',
    sub_name varchar(200) null comment '标的名称，销售、采购类型有此字段',
    delivery_method varchar(5) null comment '交货约定，销售、采购类型有此字段',
    supplier_settle_way varchar(5) null comment '付款约定，采购类型有此字段',
    merchant_pay_type varchar(5) null comment '付款约定，销售类型有此字段',
    agree_time date null comment '履约时间约定，销售、采购类型有此字段',
    stop_time date null comment '终止时间',
    recovery_time date null comment '恢复时间',
    related_contract_id bigint null comment '关联合同ID',
    contract_money decimal(12,2) null comment '合同金额',
    is_use char default '0' null comment '是否使用 0-未使用 1-已使用',
    has_sa char default '0' null comment '是否有补充协议  0-否 1-是',
    sa_pur_sale_num decimal(18,6) null comment '补充协议购销数量',
    sa_unit_price decimal(12,2) null comment '补充协议单价',
    sa_total_money decimal(12,2) null comment '补充协议的协议金额',
    audit_status varchar(30) default '1' null comment '审核状态',
    del_flag char default '0' null comment '删除状态 0-正常 2-删除',
    create_time datetime default CURRENT_TIMESTAMP null comment '创建时间',
    create_by bigint null comment '创建人',
    update_time datetime null on update CURRENT_TIMESTAMP comment '更新时间',
    update_by bigint null comment '更新人',
    tenant_id varchar(20) default '000000' null comment '租户编号',
    create_dept bigint null comment '创建部门',
    summary_id varchar(30) null comment '审批流程id'
) comment '合同' charset = utf8mb4;'''

pattern = re.compile(
    r"^\s*(\w+)\s+([a-zA-Z]+)(?:\((\d+)(?:,\s*(\d+))?\))?"  # 列名、类型、长度、精度
    r"(?:[^']*?)"                             # 跳过其他可选内容
    r"(?:not\s+null|null)?"                  # 可选：是否为空
    r"(?:\s+comment\s+'([^']*)')?",         # 可选：注释
    re.IGNORECASE
)

fields = []
for line in sql.splitlines():
    line = line.strip().rstrip(',')
    if not line or line.startswith('create ') or line.startswith(')'):
        continue
    m = pattern.search(line)
    if not m:
        continue
    name, dtype, length, precision, comment = m.groups()
    nullable = 'N' if 'not null' in line.lower() else 'Y'
    fields.append([name, dtype, length or '', precision or '', comment or '', nullable])

# 输出 CSV 文件
with open('合同表字段结构.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['列名', '类型', '长度', '精度', '注释', '是否为空'])
    writer.writerows(fields)

# 控制台打印结果
for row in fields:
    print(row)
