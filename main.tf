data "azurerm_client_config" "current" {}

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "=3.106.0"

    }
  }
}

# Configure the Microsoft Azure Provider
provider "azurerm" {
    subscription_id = "0de120ed-f2b8-4aec-84d1-474c0b8fdc77"
    tenant_id = "ad78d191-1044-4303-8212-b6f4dd7874bc"
    
    features {}
}

resource "azurerm_resource_group" "myrg" {
    name = "terraform-group5"
    location = "West Europe"

}

resource "azurerm_application_insights" "ap" {
  name                = "terraform-app"
  location            = azurerm_resource_group.myrg.location
  resource_group_name = azurerm_resource_group.myrg.name
  application_type    = "web"
}

resource "azurerm_key_vault" "keyvault" {
  name                = "terraform-keyvault-new"
  location            = azurerm_resource_group.myrg.location
  resource_group_name = azurerm_resource_group.myrg.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "premium"
}

resource "azurerm_storage_account" "sa" {
  name                     = "mlterraformsa"
  location                 = azurerm_resource_group.myrg.location
  resource_group_name      = azurerm_resource_group.myrg.name
  account_tier             = "Standard"
  account_replication_type = "GRS"
}

# Define Azure ML resources
resource "azurerm_machine_learning_workspace" "mlws" {
  name                    = "ml-workspace"
  location                = azurerm_resource_group.myrg.location
  resource_group_name     = azurerm_resource_group.myrg.name
  application_insights_id = azurerm_application_insights.ap.id
  key_vault_id            = azurerm_key_vault.keyvault.id
  storage_account_id      = azurerm_storage_account.sa.id

  public_network_access_enabled = true  # Enable public network access

  identity {
    type = "SystemAssigned"
  }
}

