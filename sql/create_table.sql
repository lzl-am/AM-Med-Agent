CREATE DATABASE IF NOT EXISTS `my_db`;
USE `my_db`;

CREATE TABLE IF NOT EXISTS drug_information (
    `id`               bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '主键id',
    product_name       VARCHAR(256) NOT NULL COMMENT '药品商品名称',
    drug_name          VARCHAR(256) NOT NULL COMMENT '药品名称',
    approval_number    VARCHAR(256) COMMENT '批准文号',
    composition        VARCHAR(512) COMMENT '成分',
    dosage_form        VARCHAR(256) COMMENT '剂型',
    specification      VARCHAR(256) COMMENT '规格',
    efficacy           VARCHAR(512) COMMENT '功效',
    dosage_and_usage   VARCHAR(512) COMMENT '用法用量',
    adverse_reactions  VARCHAR(1024) COMMENT '不良反应',
    precautions        VARCHAR(1024) COMMENT '注意事项',
    interactions       VARCHAR(1024) COMMENT '相互作用',
    therapeutic_effect VARCHAR(1024) COMMENT '疗效',
    packaging          VARCHAR(256) COMMENT '药品包装',
    manufacturer       VARCHAR(256) COMMENT '制药公司',
    category           VARCHAR(256) COMMENT '药品分类（例如：非处方药物（甲类））',
    indication         VARCHAR(1024) COMMENT '适应症/功能主治',
    create_time        TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    update_time        TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`)

) ENGINE=InnoDB
  DEFAULT CHARSET = utf8mb4 COMMENT ='药品信息表';

-- 创建索引以提高查询效率
CREATE INDEX idx_product_name ON drug_information (product_name);
CREATE INDEX idx_drug_name ON drug_information (drug_name);
